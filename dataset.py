from selenium import webdriver

# Function to save text to a file
def save_text_to_file(text, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)

# Initialize the WebDriver without specifying the executable path
driver = webdriver.Chrome('D:/resumeai\chromedriver_win32\chromedriver.exe')

# Navigate to the web page you want to scrape
driver.get('https://google.com')  # Replace with the URL of the web page

# Wait for the page to load (you may need to adjust the time)
driver.implicitly_wait(10)

# Extract the page content as text
page_content = driver.page_source

# Save the content to a text file
save_text_to_file(page_content, 'web_page_content.txt')

# Close the browser
driver.quit()

# Function to filter keywords from a text file
def filter_keywords(text_file, keywords):
    with open(text_file, 'r', encoding='utf-8') as file:
        text = file.read()

    # Filter keywords from the text
    filtered_text = [keyword for keyword in keywords if keyword.lower() in text.lower()]

    return filtered_text

if __name__ == "__main__":
    # Define keywords to filter
    keywords = ['keyword1', 'keyword2', 'keyword3']  # Replace with your desired keywords

    # Filter keywords from the text file
    filtered_keywords = filter_keywords('web_page_content.txt', keywords)

    if filtered_keywords:
        print("Keywords found in the web page content:")
        for keyword in filtered_keywords:
            print(keyword)
    else:
        print("No keywords found in the web page content.")
