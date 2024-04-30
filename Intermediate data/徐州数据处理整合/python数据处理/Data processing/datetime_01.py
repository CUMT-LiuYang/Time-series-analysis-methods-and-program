import datetime

start_date = datetime.datetime(2024, 1, 1, 0, 0, 0)
end_date = datetime.datetime(2024, 4, 6, 23, 0, 0)

current_date = start_date
while current_date <= end_date:
    formatted_date = current_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    print(formatted_date)
    current_date += datetime.timedelta(hours=1)
