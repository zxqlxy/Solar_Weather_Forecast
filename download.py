import requests
from datetime import datetime, timedelta

from config import start, end

start_time = datetime.strptime(start, '%Y-%m-%d %H:%M')
end_time = datetime.strptime(end, '%Y-%m-%d %H:%M')
delta = timedelta(minutes = 2)
urlBase = "http://jsoc.stanford.edu/data/aia/synoptic"
locPath = "/Users/lxy/Desktop/Rice/Su 2021/Solar_Weather_Forecast/data/SDOAIA"

time = start_time
while time < end_time:
    time += delta
    yr = str(time.year)
    mo = str(time.month) if time.month >= 10 else '0' + str(time.month)
    da = str(time.day) if time.day >= 10 else '0' + str(time.day)
    ho = str(time.hour) if time.hour >= 10 else '0' + str(time.hour)
    mi = str(time.minute) if time.minute >= 10 else '0' + str(time.minute)
    thisFile = "AIA"+ yr + mo + da + "_" + ho + mi + "_0094.fits"
    thisPath = "/".join([yr,mo,da,"H" + ho + "00"])
    print(thisFile)

    myUrl = "/".join([urlBase,thisPath,thisFile ])
    myDest = "/".join([locPath, thisFile])
    # print(myDest)
    print(myUrl)
    myBits = requests.get(myUrl)

    if len(myBits.content) > 500: # if files don't exist we get a short reply from the server: skip these
        myFile = open(myDest, "wb")
        myFile.write(myBits.content)
        myFile.close()
