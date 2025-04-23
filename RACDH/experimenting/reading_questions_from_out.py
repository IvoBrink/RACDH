import json

all_data = []
current_data = {}

in_alice_bob_section = False
in_question_section = False
skip_until_truncate = False

with open('RACDH/job_outputs/data_generation/knowledge_tests.out', 'r') as file:
    for line in file:
        line_stripped = line.strip()

        # Detect a new title marker
        if "----------------------------------------" in line:
            title = line.replace("-", "").strip()

        # If we encounter a line that starts a new "Generating know-tests for" block,
        # we first store any existing data, then reset current_data.
        if "-------------------- Generating know-tests for" in line:
            if current_data:
                all_data.append(current_data)
            current_data = {}
            current_data["title"] = title
            current_data["entity"] = (
                line.replace("-", "")
                .replace("Generating knowtests for", "")
                .strip()
            )

        # Detect the start of the Alice-Bob conversation section
        if "---------- Alice-bob generation ----------" in line:
            current_data["alice_bob_conversation"] = []
            i = 0
            while True:
                line = next(file)
                if "----- Truncate entity -----" in line or i > 10:
                    break
                i += 1
            
            alice_bob = []
            for _ in range(3):
                alice_bob.append(next(file))

            current_data["alice_bob_conversation"] = "".join(alice_bob)[:-1]

        # Detect the start of "Question generation" section
        if "---------- Question generation ----------" in line:
            current_data["question"] = []
            i = 0
            while True:
                line = next(file)
                if "----- Truncate entity -----" in line   or i > 10:
                    break
                i += 1
            
            question = []
            for _ in range(2):
                question.append(next(file))

            current_data["question"] = "".join(question)[:-1]
      

        # Detect the "Truncate passage" marker
        if "---------- Truncate passage 'till entity ----------" in line:
            # Get the next line
            current_data["truncated_passage"] = next(file).strip()

    # After reading all lines, if there's one last piece of current_data, append it
    if current_data:
        all_data.append(current_data)

# Finally, save to JSON
with open('knowledge_questions.json', 'w') as json_file:
    json.dump(all_data, json_file, indent=4)

print(json.dumps(all_data, indent=4))
