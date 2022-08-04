import os

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

print(os.getenv('SGIPSIGNAL_USER'))
print(os.getenv('SGIPSIGNAL_PASS'))