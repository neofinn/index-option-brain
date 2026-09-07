//+------------------------------------------------------------------+
//|  IndexBrainSignals.mq5                                            |
//|  Polls the signal gateway and reconciles the position it names.    |
//+------------------------------------------------------------------+
//
//  WHY AN EA POLLS RATHER THAN RECEIVES
//  MetaTrader has no listening socket. It cannot be sent a webhook, so
//  the gateway holds relayed signals and this EA reads them on a timer
//  with WebRequest. That is why the route's destination is "pull": the
//  relay's job for an EA is to hold, not to deliver.
//
//  BEFORE IT WILL RUN
//  Tools -> Options -> Expert Advisors -> "Allow WebRequest for listed
//  URL" and add the gateway's base URL exactly, e.g.
//      https://hooks.example.com
//  WebRequest returns -1 with error 4014 until you do. There is no way
//  around it from inside the EA, and that is deliberate on MetaQuotes'
//  part.
//
//  IT RECONCILES A TARGET, IT DOES NOT REPLAY DELTAS
//  Where the signal carries a target position, this EA works out the
//  difference between that target and what the account actually holds,
//  and trades the difference. Replaying "buy 2 / sell 1" deltas is the
//  obvious design and the wrong one: webhooks are at-most-once, so one
//  missed signal leaves a delta-replaying EA permanently out of step with
//  the chart, and it stays wrong for every trade after it. Reconciling a
//  target self-heals on the next signal.
//
//  Written for a NETTING account, where one symbol has one net position.
//  On a hedging account PositionSelect returns only the first position
//  for the symbol, so the arithmetic below is wrong — set the account to
//  netting, or replace CurrentNetVolume() with a loop over PositionsTotal.
//
//  THIS CODE HAS NOT BEEN RUN AGAINST A BROKER
//  EnableTrading is false by default and every decision is printed
//  instead. Run it on a demo account, read the log, and only then
//  consider turning it on. An EA that places orders is not something to
//  trust because it compiled.
//
//  MQL4: WebRequest and StringSplit are the same; the position and order
//  functions are not. The polling and parsing half ports unchanged; the
//  Execute() half needs rewriting against OrderSend/OrderSelect.
//+------------------------------------------------------------------+
#property copyright "Index Option Brain"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//--- connection
input string  GatewayBase   = "https://hooks.example.com";  // no trailing slash
input string  RouteSlug     = "ea";                          // the gateway route
input string  ReadToken     = "";                            // the route's read_token
input int     PollSeconds   = 5;
input int     RequestTimeout= 5000;                          // milliseconds

//--- safety
input bool    EnableTrading = false;   // false = log the decision, place nothing
input double  MaxVolume     = 1.0;     // per-order ceiling, in lots
input double  VolumeScale   = 1.0;     // signal quantity -> lots
input int     Slippage      = 20;      // points
input long    MagicNumber   = 20260907;

//--- state
CTrade  trade;
string  cursorVar;      // terminal global variable holding the cursor
int     consecutiveErrors = 0;

//+------------------------------------------------------------------+
int OnInit()
  {
   if(StringLen(ReadToken) < 16)
     {
      Print("IndexBrainSignals: ReadToken is missing or too short. ",
            "Paste the route's read_token from the gateway page.");
      return(INIT_PARAMETERS_INCORRECT);
     }
   if(StringFind(GatewayBase, "https://") != 0 &&
      StringFind(GatewayBase, "http://127.0.0.1") != 0)
     {
      // The token travels in a header. Plain http to a remote host puts
      // it on the wire in clear text.
      Print("IndexBrainSignals: GatewayBase must be https, or a loopback ",
            "address for a local bridge.");
      return(INIT_PARAMETERS_INCORRECT);
     }

   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(Slippage);
   trade.SetTypeFillingBySymbol(_Symbol);

   // The cursor is keyed on the route so two EAs on two routes do not
   // consume each other's signals.
   cursorVar = "IBS_cursor_" + RouteSlug;
   if(!GlobalVariableCheck(cursorVar))
      GlobalVariableSet(cursorVar, 0.0);

   EventSetTimer(MathMax(1, PollSeconds));
   PrintFormat("IndexBrainSignals: route=%s trading=%s cursor=%d",
               RouteSlug,
               EnableTrading ? "LIVE" : "disabled (logging only)",
               (long)GlobalVariableGet(cursorVar));
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
  }

//+------------------------------------------------------------------+
void OnTimer()
  {
   Poll();
  }

//+------------------------------------------------------------------+
//| One poll: fetch, parse, act, advance the cursor.                  |
//| The cursor is advanced only after the whole body has been handled,|
//| so a parse failure re-reads the same window next tick rather than |
//| skipping past a signal nobody acted on.                           |
//+------------------------------------------------------------------+
void Poll()
  {
   long cursor = (long)GlobalVariableGet(cursorVar);
   string url = StringFormat("%s/v1/%s/signals.csv?since=%d&limit=50",
                             GatewayBase, RouteSlug, cursor);
   string headers = "Authorization: Bearer " + ReadToken + "\r\n";

   char post[], result[];
   string resultHeaders;
   ArrayResize(post, 0);

   ResetLastError();
   int status = WebRequest("GET", url, headers, RequestTimeout,
                           post, result, resultHeaders);
   if(status == -1)
     {
      int err = GetLastError();
      consecutiveErrors++;
      if(err == 4014)
         Print("IndexBrainSignals: add ", GatewayBase,
               " to Tools -> Options -> Expert Advisors -> Allow WebRequest.");
      else if(consecutiveErrors == 1 || consecutiveErrors % 12 == 0)
         // Not on every tick: a gateway that is down for an hour would
         // otherwise write 720 identical lines into the log.
         PrintFormat("IndexBrainSignals: WebRequest failed (error %d, %d in a row)",
                     err, consecutiveErrors);
      return;
     }
   if(status != 200)
     {
      consecutiveErrors++;
      PrintFormat("IndexBrainSignals: gateway answered %d", status);
      return;
     }
   consecutiveErrors = 0;

   string body = CharArrayToString(result, 0, WHOLE_ARRAY, CP_UTF8);
   string lines[];
   int count = StringSplit(body, '\n', lines);
   if(count < 1)
      return;

   long nextCursor = cursor;
   for(int i = 0; i < count; i++)
     {
      string line = lines[i];
      StringTrimLeft(line);
      StringTrimRight(line);
      if(StringLen(line) == 0)
         continue;
      if(StringFind(line, "#cursor=") == 0)
        {
         nextCursor = (long)StringToInteger(StringSubstr(line, 8));
         continue;
        }
      if(StringFind(line, "#error") == 0)
        {
         Print("IndexBrainSignals: ", line);
         return;
        }
      if(StringFind(line, "seq,") == 0)
         continue;               // the header row
      HandleRow(line);
     }

   if(nextCursor != cursor)
      GlobalVariableSet(cursorVar, (double)nextCursor);
  }

//+------------------------------------------------------------------+
//| Columns, fixed by the gateway and read by index:                  |
//| 0 seq, 1 fired_at, 2 strategy, 3 symbol, 4 action,                |
//| 5 quantity, 6 target, 7 price, 8 order_id                          |
//+------------------------------------------------------------------+
void HandleRow(const string line)
  {
   string f[];
   int n = StringSplit(line, ',', f);
   if(n < 7)
     {
      Print("IndexBrainSignals: short row ignored: ", line);
      return;
     }

   long   seq      = (long)StringToInteger(f[0]);
   string strategy = f[2];
   string symbol   = f[3];
   string action   = f[4];
   double quantity = StringToDouble(f[5]);
   string targetTx = f[6];

   if(!SymbolSelect(symbol, true))
     {
      // The route maps the TradingView ticker to this name. An unmapped
      // or misspelled one must not fall through to the chart's symbol —
      // that would trade the wrong instrument with a plausible size.
      PrintFormat("IndexBrainSignals: #%d %s names symbol %s, which this "
                  "terminal does not have. Fix the route's symbol_map.",
                  seq, strategy, symbol);
      return;
     }

   double target;
   bool haveTarget = (StringLen(targetTx) > 0);
   if(action == "exit")
     {
      target = 0.0;             // fully specified without a quantity
      haveTarget = true;
     }
   else if(haveTarget)
      target = StringToDouble(targetTx) * VolumeScale;
   else
     {
      // Delta only. Applied to what is actually held, so the EA still
      // ends up reconciling a position rather than firing blind.
      double current = CurrentNetVolume(symbol);
      double signed_ = (action == "buy" ? quantity : -quantity) * VolumeScale;
      target = current + signed_;
      haveTarget = true;
     }

   Execute(seq, strategy, symbol, target);
  }

//+------------------------------------------------------------------+
//| Net signed volume held for a symbol. Positive long, negative      |
//| short, zero flat. Netting accounts only — see the header.         |
//+------------------------------------------------------------------+
double CurrentNetVolume(const string symbol)
  {
   if(!PositionSelect(symbol))
      return(0.0);
   double volume = PositionGetDouble(POSITION_VOLUME);
   long   type   = PositionGetInteger(POSITION_TYPE);
   return(type == POSITION_TYPE_SELL ? -volume : volume);
  }

//+------------------------------------------------------------------+
//| Trade the difference between what is held and what is wanted.     |
//+------------------------------------------------------------------+
void Execute(const long seq, const string strategy,
             const string symbol, const double target)
  {
   double current = CurrentNetVolume(symbol);
   double delta   = target - current;

   double step = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   if(step <= 0.0)
      step = 0.01;
   // Rounded to the broker's step before the "is it worth doing" test,
   // so a residual smaller than one step is treated as flat rather than
   // retried on every poll forever.
   delta = MathRound(delta / step) * step;

   if(MathAbs(delta) < step / 2.0)
     {
      PrintFormat("IndexBrainSignals: #%d %s %s already at %.2f — nothing to do",
                  seq, strategy, symbol, target);
      return;
     }
   if(MathAbs(delta) > MaxVolume)
     {
      PrintFormat("IndexBrainSignals: #%d %s %s wants %.2f lots, over MaxVolume "
                  "%.2f — refused", seq, strategy, symbol, MathAbs(delta), MaxVolume);
      return;
     }

   if(!EnableTrading)
     {
      PrintFormat("IndexBrainSignals: [dry run] #%d %s %s %s %.2f "
                  "(held %.2f -> target %.2f)",
                  seq, strategy, symbol, delta > 0 ? "BUY" : "SELL",
                  MathAbs(delta), current, target);
      return;
     }

   bool ok = false;
   if(target == 0.0 && current != 0.0)
      ok = trade.PositionClose(symbol);
   else if(delta > 0.0)
      ok = trade.Buy(MathAbs(delta), symbol);
   else
      ok = trade.Sell(MathAbs(delta), symbol);

   if(ok)
      PrintFormat("IndexBrainSignals: #%d %s %s %s %.2f done (retcode %d)",
                  seq, strategy, symbol, delta > 0 ? "BUY" : "SELL",
                  MathAbs(delta), trade.ResultRetcode());
   else
      // Not retried. A broker call whose result is unknown may well have
      // been executed, and a blind retry is how one signal becomes two
      // positions. The next signal restates the target and reconciles.
      PrintFormat("IndexBrainSignals: #%d %s %s FAILED retcode=%d %s",
                  seq, strategy, symbol, trade.ResultRetcode(),
                  trade.ResultRetcodeDescription());
  }
//+------------------------------------------------------------------+
