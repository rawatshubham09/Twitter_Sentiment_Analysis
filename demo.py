from twitter_sentiment.logger import logging
import sys

from twitter_sentiment.exception import TwetterException

logging.info("welcome in custome logger")


try:
    a=2/0

except Exception as e:
    raise TwetterException(e,sys)