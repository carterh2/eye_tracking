def yes_no_prompt(prompt, func):
  while True:
    # put prompting in game loop to prevent typos from crashing the programm
    answer = input(f"{prompt}[y,N]:")

    if answer == "y":
        func()
        break
    elif answer == "N":
        break
    else:
        print("Invalid answer")