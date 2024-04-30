import datetime

start_date = datetime.datetime(2023, 1, 1, 0, 0, 0)
end_date = datetime.datetime(2023, 12, 31, 23, 0, 0)

with open("output.txt", "w") as f:
    current_date = start_date
    while current_date <= end_date:
        formatted_date = current_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        f.write(formatted_date + "\n")
        current_date += datetime.timedelta(hours=1)
