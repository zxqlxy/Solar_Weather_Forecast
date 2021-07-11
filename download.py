"""
Use python concurrent to download files in batches, considerably save the time
"""

import concurrent
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import requests
from datetime import datetime, timedelta
from os import path
import time as t

from config import start, end, delta, wavelength, locPath

start_time = datetime.strptime(start, '%Y-%m-%d %H:%M')
end_time = datetime.strptime(end, '%Y-%m-%d %H:%M')
urlBase = "http://jsoc.stanford.edu/data/aia/synoptic"
count = (end_time - start_time) // delta            # number of data points

def download(myUrl, myDest):
    # Skip the file if downloaded
    if path.isfile(myDest):
        return "file downloaded"
    print(myUrl)
    myBits = requests.get(myUrl)
    t.sleep(1)
    if myBits.status_code == 404:
        return "file not exists"

    if len(myBits.content) > 500: # if files don't exist we get a short reply from the server: skip these
        myFile = open(myDest, "wb")
        myFile.write(myBits.content)
        myFile.close()
        return "SUCCEED"
    else:
        return "file downlaod timed out"

def parellel(wavelength):
    URLS = []
    DEST = []

    # First get all the urls
    time = start_time
    while time < end_time:
        time += delta
        yr = str(time.year)
        mo = str(time.month) if time.month >= 10 else '0' + str(time.month)
        da = str(time.day) if time.day >= 10 else '0' + str(time.day)
        ho = str(time.hour) if time.hour >= 10 else '0' + str(time.hour)
        mi = str(time.minute) if time.minute >= 10 else '0' + str(time.minute)
        thisFile = "AIA"+ yr + mo + da + "_" + ho + mi + "_0" + wavelength + ".fits"
        thisPath = "/".join([yr,mo,da,"H" + ho + "00"])
        # print(thisFile)

        myUrl = "/".join([urlBase,thisPath,thisFile])
        myDest = "\\".join([locPath, thisFile])

    # 5 is a number that won't introduce max tries exceed error
    with ThreadPoolExecutor(max_workers = 10) as executor: 
        # Start the load operations and mark each future with its URL
        future_to_url = {executor.submit(download, URLS[i], DEST[i]): i for i in range(count)}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (url, exc))
            else:
                print('%r : %s' % (url, data))


if __name__ == "__main__":
    import sys
    print(sys.argv)
    wave = sys.argv[1]
    parellel(wave)