import os
import time
import random
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--log-level=3")  # Suppress logs
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920x1080")
chrome_options.add_argument("--headless")  # Run in headless mode

# Initialize the Chrome driver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

def scroll_to_load_images(scroll_times=50, wait_time_range=(2, 5)):
    """Scrolls the page multiple times to load more images with random wait time."""
    for _ in range(scroll_times):
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
        time.sleep(random.uniform(*wait_time_range))  # Random sleep time

def download_images(search_query, num_images=500):
    """Scrapes and downloads images from Google."""
    
    search_url = f"https://www.google.com/search?q={search_query}&tbm=isch"
    driver.get(search_url)
    
    scroll_to_load_images(scroll_times=50)  # Increase scroll depth

    # Find image elements
    image_elements = driver.find_elements(By.CSS_SELECTOR, "img")

    if len(image_elements) < num_images:
        print(f"⚠ Warning: Found only {len(image_elements)} images for {search_query}. Trying more scrolling...")
        scroll_to_load_images(scroll_times=40)  # Extra scrolling
        image_elements = driver.find_elements(By.CSS_SELECTOR, "img")  # Refresh list

    # Create folder for images
    folder_name = f"dataset/{search_query.replace(' ', '_')}"
    os.makedirs(folder_name, exist_ok=True)

    count = 0
    for img in image_elements:
        if count >= num_images:
            break

        img_url = img.get_attribute("src") or img.get_attribute("data-src")
        if img_url and img_url.startswith("http"):  # Ensure valid URL
            try:
                img_data = requests.get(img_url, timeout=10).content  # Increased timeout
                with open(f"{folder_name}/{count}.jpg", "wb") as img_file:
                    img_file.write(img_data)
                count += 1
                print(f"📸 Downloaded {count}/{num_images} images for {search_query}")
                time.sleep(random.uniform(1, 4))  # Random wait to avoid bot detection
            except Exception as e:
                print(f"❌ Error downloading image {count}: {e}")

    print(f"✅ Finished downloading {count} images for {search_query}")

# 🏆 List of sports equipment to scrape
sports_equipment = [
    "Badminton_Racket", "Baseball_ball", "Baseball_Bat", "Basketball_ball",
    "Billiard_Cue", "Bow_and_Arrow_Archery", "Boxing_Gloves", "Carrom_Board",
    "Carrom_Coins", "Chess_Board", "Cricket_Ball", "Cricket_Bat", 
    "Hockey_Ball", "Hockey_Stick", "Shuttlecock", "Skateboard", 
    "Soccer_Ball", "Squash_Racket", "Table_Tennis_Ball", "Table_Tennis_Paddle", 
    "Tennis_Ball", "Tennis_Racket",  "Volleyball_ball"      
]


# Loop through each equipment and scrape 500 images
for item in sports_equipment:
    download_images(item, num_images=500)

# Close the browser
driver.quit()


# 🏆 List of sports equipment to scrape
#sports_equipment = [
#    "Badminton_Racket", "Tennis_Racket", "Table_Tennis_Paddle", "Squash_Racket",
#    "Shuttlecock", "Tennis_Ball", "Table_Tennis_Ball", "Soccer_Ball", "Basketball_ball",
#    "Volleyball_ball", "Cricket_Ball", "Baseball_ball", "Hockey_Ball",
#    "Cricket_Bat", "Baseball_Bat", "Hockey_Stick", "Chess_Board",
#    "Carrom_Board", "Carrom_Coins", "Bow_and_Arrow_Archery",
#    "Billiard_Cue", "Skateboard", "Boxing_Gloves"
#]