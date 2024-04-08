import os
import datetime
import time


def get_timestamp_now():
    return str(int(time.time()*1000))


def get_oldest_date(file_path):
    # Get the file metadata
    file_stats = os.stat(file_path)

    # Get the timestamps
    creation_time = datetime.datetime.fromtimestamp(file_stats.st_ctime)
    access_time = datetime.datetime.fromtimestamp(file_stats.st_atime)
    modification_time = datetime.datetime.fromtimestamp(file_stats.st_mtime)

    # Find the oldest date
    oldest_date = min(creation_time, access_time, modification_time)

    # Convert the oldest date to milliseconds since the epoch
    epoch = datetime.datetime.utcfromtimestamp(0)
    delta = oldest_date - epoch
    return int(delta.total_seconds() * 1000)
