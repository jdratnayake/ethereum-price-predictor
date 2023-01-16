import pandas as pd
import yfinance as yf
from datetime import datetime
from datetime import timedelta
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import warnings
import plotly.graph_objects as go

warnings.filterwarnings('ignore')
pd.options.display.float_format = '${:,.2f}'.format

# define ranges
today = datetime.today().strftime('%Y-%m-%d')
start_date = '2016-01-01'

# get ethereum data
eth_df = yf.download('ETH-USD', start_date, today)
eth_df.reset_index(inplace=True)

df = eth_df[["Date", "Open"]]
rename_columns = {
    "Date":"ds",
    "Open":"y"
}
df.rename(columns=rename_columns, inplace=True)

x = df["ds"]
y = df["y"]

# plot the open price
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y))
fig.update_layout(
title_text="ime series plot of Ethereum Open Price"
)
fig.write_image("open_price_plot.png")

m = Prophet(seasonality_mode="multiplicative")
# remove timezone information
df["ds"] = df["ds"].dt.tz_localize(None)
m.fit(df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

next_day = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
print("\n%s" % next_day)
print(forecast[forecast["ds"]==next_day]["yhat"].item())

# plot the forecast data
fig2 = plot_plotly(m, forecast)
fig2.write_image("forecast_plot.png")

fig3 = plot_components_plotly(m, forecast)
fig3.write_image("forecast_component_plot.png")