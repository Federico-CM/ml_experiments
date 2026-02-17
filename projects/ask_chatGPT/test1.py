from openai import OpenAI
import sys

# Default placeholders
my_key = "MY API KEY"  # You can write your API key here
my_prompt = "Say Hello world, this is ChatGPT"  # add your prompt here

# Check if API key is still placeholder
if my_key == "MY API KEY":
    print("An API key is required.")
    print("You can get one at:")
    print("https://platform.openai.com/settings/organization/api-keys")
    print("\nInsert your API key to continue or press 'a' to abort.")

    user_input = input("API Key (or 'a' to abort): ").strip()

    if user_input.lower() == "a":
        print("Aborted by user.")
        sys.exit()

    if not user_input:
        print("No API key provided. Exiting.")
        sys.exit()

    my_key = user_input

# Check if prompt is still default placeholder
if my_prompt == "Say Hello world, this is ChatGPT":
    print("\nNo custom prompt detected.")
    user_prompt = input("Please enter your prompt (or press 'a' to abort): ").strip()

    if user_prompt.lower() == "a":
        print("Aborted by user.")
        sys.exit()

    if not user_prompt:
        print("No prompt provided. Exiting.")
        sys.exit()

    my_prompt = user_prompt

# Create client
client = OpenAI(api_key=my_key)

def ask_chatgpt():
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": my_prompt}],
    )
    return response.choices[0].message.content.strip()

def main():
    answer = ask_chatgpt()
    print(answer)

if __name__ == "__main__":
    main()


