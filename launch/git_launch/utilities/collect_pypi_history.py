from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup


def find_latest_version(package_name, query_date):
    date_version_mapping = collect_pypi_history(package_name)
    query_date = datetime.fromisoformat(query_date).replace(tzinfo=timezone.utc)
    # find the latest version before the query date
    if not date_version_mapping:
        return None
    latest_version = None
    for date, version in date_version_mapping:
        if date < query_date:
            latest_version = version
            break
    return latest_version


def collect_pypi_history(package_name):
    url = f"https://pypi.org/project/{package_name}/#history"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch data for package: {package_name}")
        return
    soup = BeautifulSoup(response.text, "html.parser")
    releases = soup.find_all("div", class_="release")
    date_version_mapping = []
    for release in releases:
        version = release.find("p", class_="release__version").text.strip()
        date = release.find("time")[
            "datetime"
        ].strip()  # Extract the 'datetime' attribute
        date = datetime.fromisoformat(date)  # Convert to datetime object
        if date.tzinfo is None:  # Ensure the date is offset-aware
            date = date.replace(tzinfo=timezone.utc)
        date_version_mapping.append((date, version))

    return date_version_mapping


if __name__ == "__main__":
    package_name = "numpy"
    history = collect_pypi_history(package_name)
    if history:
        for date, version in history:
            print(f"Date: {date}, Version: {version}")

    latest_version = find_latest_version("numpy", "2023-10-01")
    print(f"Latest version before 2023-10-01: {latest_version}")
