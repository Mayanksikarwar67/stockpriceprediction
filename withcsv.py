import csv
import re
import os
import time
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# Explicit paths for saving files and ChromeDriver
SAVE_DIR = "/Users/mayanksikarwar/Desktop/stock_price_prediction"
CHROME_DRIVER_PATH = "/Users/mayanksikarwar/Downloads/chromedriver-mac-arm64/chromedriver"

# Function to convert a date string to Unix timestamp
def date_to_timestamp(date_str):
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    return int(time.mktime(date_obj.timetuple()))

# Function to generate the Yahoo Finance link
def generate_yahoo_finance_link(stock_code, start_date, end_date):
    start_timestamp = date_to_timestamp(start_date)
    end_timestamp = date_to_timestamp(end_date)
    return (f"https://finance.yahoo.com/quote/{stock_code}/history/"
            f"?filter=history&frequency=1d&period1={start_timestamp}&period2={end_timestamp}")

# Function to save the HTML of a page
def save_page_html(stock_code, start_date, end_date, save_dir):
    url = generate_yahoo_finance_link(stock_code, start_date, end_date)
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    service = Service(CHROME_DRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.get(url)
    driver.implicitly_wait(5)
    html_content = driver.page_source
    file_path = os.path.join(save_dir, f"{stock_code}_{start_date}_to_{end_date}.html")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(html_content)
    driver.quit()
    return file_path

# Function to wait for the specific HTML file
def wait_for_file(filename):
    while not os.path.exists(filename):
        time.sleep(1)
    print(f"File {filename} found!")
    return filename

# Function to convert HTML to text
def html_to_text(html_file):
    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    text_content = soup.get_text("\n", strip=True)
    txt_filename = os.path.splitext(html_file)[0] + ".txt"
    with open(txt_filename, "w", encoding="utf-8") as txt_file:
        txt_file.write(text_content)
    return txt_filename

# Function to extract stock data and save to CSV
def extract_stock_data_to_csv(file_path, output_csv_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        print("Failed to read file with UTF-8 encoding. Trying ISO-8859-1...")
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            content = file.read()

    pattern = re.compile(
        r'(\w{3} \d{1,2}, \d{4})\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)\s+([\d,]+)'
    )
    matches = pattern.findall(content)
    if not matches:
        print("No stock data found in the file.")
        return

    headers = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    rows = [
        [date, open_, high, low, close, adj_close, volume.replace(',', '')] 
        for date, open_, high, low, close, adj_close, volume in matches
    ]

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Stock data has been saved to {output_csv_path}")

    html_file = os.path.splitext(file_path)[0] + ".html"
    txt_file = os.path.splitext(file_path)[0] + ".txt"
    
    if os.path.exists(html_file):
        os.remove(html_file)
        print(f"Deleted file: {html_file}")
    if os.path.exists(txt_file):
        os.remove(txt_file)
        print(f"Deleted file: {txt_file}")

# Main function
def automate_yahoo_finance_scraping(stock_code, start_date, end_date):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    html_file = save_page_html(stock_code, start_date, end_date, SAVE_DIR)
    wait_for_file(html_file)
    
    output_csv_path = os.path.join(SAVE_DIR, f"{stock_code}_{start_date}_to_{end_date}.csv")
    text_file = html_to_text(html_file)
    extract_stock_data_to_csv(text_file, output_csv_path)

    print(f"The stock data has been successfully saved to: {output_csv_path}")

# Example usage
stock_code = input("Enter the stock ticker symbol (e.g., AAPL): ").strip().upper()
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today().replace(year=datetime.today().year - 5)).strftime('%Y-%m-%d')
automate_yahoo_finance_scraping(stock_code, start_date, end_date)
