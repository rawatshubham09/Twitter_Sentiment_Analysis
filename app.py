
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional
from twitter_sentiment.logger import logging

from twitter_sentiment.constants import APP_HOST, APP_PORT
from twitter_sentiment.pipline.prediction_pipeline import TweetsData, TweetsClassifier
from twitter_sentiment.pipline.training_pipeline import TrainPipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.tweet: Optional[str] = None
        

    async def get_tweet_data(self):
        form = await self.request.form()
        self.tweet = form.get("tweet")
        

@app.get("/", tags=["authentication"])
async def index(request: Request):

    return templates.TemplateResponse(
            "tweets.html",{"request": request, "context": "Rendering"})


#@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/")
async def predictRouteClient(request: Request):
    try:
        logging.info("Entered  app.py predictRouteClient")
        form = DataForm(request)
        await form.get_tweet_data()
        
        tweets_data = TweetsData(tweet= form.tweet)
        
        tweets_df = tweets_data.get_tweets_input_data_frame()

        logging.info("Get Dataframe now calling model predictor TweetClassifier")

        model_predictor = TweetsClassifier()

        value = model_predictor.predict(dataframe=tweets_df)[0]

        status = None
        if value == 1:
            status = "Positive"
        else:
            status = "Negative"

        return templates.TemplateResponse(
            "tweets.html",
            {"request": request, "context": status},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)