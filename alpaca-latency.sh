# Retrieve API key and secret from environment variables.
API_KEY=$ALPACA_API_KEY
API_SECRET=$ALPACA_SECRET_KEY

# Check if the keys are set
if [ -z "$API_KEY" ] || [ -z "$API_SECRET" ]; then
    echo "API key or Secret key is not set in environment variables."
    exit 1
fi 
# Perform latency test using curl

curl -o /dev/null -s -w "\nTime: %{time_total}seconds to retrieve 10,000 SPY trades from 2024-10-11. \n" --request GET --url 'https://data.alpaca.markets/v2/stocks/SPY/trades?start=2024-10-11&end=2024-10-11&limit=10000&feed=sip&sort=asc' --header "APCA-API-KEY-ID: $ALPACA_API_KEY" --header "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY" --header 'accept: application/json'

curl -o /dev/null -s -w "\nTime: %{time_total}seconds to retrieve max SPY 1-Min-bars from 2024-10-11. \n" --request GET --url 'https://data.alpaca.markets/v2/stocks/SPY/bars?timeframe=1Min&start=2024-10-11&end=2024-10-11&limit=10000&adjustment=raw&feed=sip&sort=asc' --header "APCA-API-KEY-ID: $ALPACA_API_KEY" --header "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY" --header 'accept: application/json'