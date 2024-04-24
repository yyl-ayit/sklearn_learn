import plotly.express as px
import plotly.graph_objects as go
df = px.data.iris()

fig = px.scatter(df, x=df.sepal_length, y=df.sepal_width)

fig.show()