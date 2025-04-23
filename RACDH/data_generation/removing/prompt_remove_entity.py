from RACDH.data_generation.instruct_model import generate_completion
from RACDH.config import params
from RACDH.data_generation.utils.print import *
import random

def remove_entity_from_passage(entity, passage):
    if params.debug: print_h2(f"Rewrite passage for [{entity}]")
    prompt, pattern = get_prompt(entity, passage)
    completion = generate_completion(prompt, pattern, max_new_tokens=516, temperature=0.5, debug=params.debug)
    if sanity_checks(entity, completion):
        return completion
    else:
        print_warning("Entity not succesfully removed from passage")
        return None


def sanity_checks(entity, completion):
    # Simple case-insensitive check to see if the entity is still present
    if entity.lower() in completion.lower():
        return False
    return True

def get_prompt(entity, passage):
    _FORBIDDEN = "another, other, elsewhere, someplace, something, someone, notable, major, region, and global."
    ex = random.choice(_EXAMPLES_REMOVE_PROMPT)

    prompt = f"""### ROLE
You are an expert copy‑editor.

### TASK
Remove every explicit mention of the entity **and any variant (abbreviation, \
nickname, unambiguous pronoun)** from the passage **while preserving factual content \
and fluency**.

### ONE ILLUSTRATIVE EXAMPLE
Entity:
<<< {ex['entity']} >>>
Original passage:
<<< {ex['original']} >>>
Rewritten passage:
<<< {ex['rewritten']} >>>

### CONSTRAINTS
If the entity can easily be omitted, without changing the any other text. Do this always. Otherwise, rewrite the text following these constraints:
1. Do **not** insert placeholder tokens such as “_____” or “[ENTITY]”.
2. **Forbidden filler words:** {_FORBIDDEN}.
If a replacement is needed, choose a context‑specific noun phrase \
(e.g. “the nation”, “the author”) or restructure the sentence.
3. If deleting the entity breaks a sentence, repair the grammar so the passage \
would pass professional copy‑edit.
4. Keep all dates, numbers and named entities that do **not** refer to the \
target entity.
5. When the entity acts as an adjective (“UK policy”), rewrite only that phrase \
so the head noun remains (“the policy”).
6. **Meta‑instruction:** Do **not** mimic specific wording from the example; use \
whatever phrasing fits the original passage.
7. **Self‑check:** After rewriting, scan the passage once more and remove any \
forbidden filler words or dangling pronouns. Make sure no double spaces remain.

### NOW YOUR TURN
Entity:
<<< {entity} >>>
Original passage:
<<< {passage} >>>
Rewritten passage (end with >>>):
<<<""".strip()

    # The regex pattern that captures the text inside Rewritten passage: <<< >>>
    pattern = r'Rewritten passage \(ending with >>>\):\s*<<<(.*?)>>>\s*(?:\})?'
    return prompt, pattern


_EXAMPLES_REMOVE_PROMPT = [
    # ─────────────────────────── EXAMPLE 1 — politics / WW2  (unchanged) ───────────────────────────
    dict(
        entity="United Kingdom",
        original=(
            "Winston Churchill was a British statesman, soldier, and writer who "
            "served as Prime Minister of the United Kingdom from 1940 to 1945 "
            "and again from 1951 to 1955. He led Britain to victory in the Second "
            "World War. Among the British public, he is widely considered the "
            "greatest Briton of all time. He was born to an aristocratic family "
            "in Oxfordshire, England."
        ),
        rewritten=(
            "Winston Churchill was a statesman, soldier, and writer who served "
            "as Prime Minister from 1940 to 1945 and again from 1951 to 1955. "
            "He led the nation to victory in the Second World War. Among the "
            "public, he is widely considered one of the greatest leaders of all "
            "time. He was born to an aristocratic family in Oxfordshire."
        ),
    ),

    # ─────────────────────────── EXAMPLE 2 — science  (remove “theory of relativity”) ───────────────────────────
    dict(
        entity="the theory of relativity",
        original=(
            "Albert Einstein was a German‑born theoretical physicist who developed "
            "the theory of relativity, one of the two pillars of modern physics. "
            "Einstein's mass–energy equivalence formula E = mc² has been dubbed "
            "\"the world's most famous equation.\""
        ),
        rewritten=(
            "Albert Einstein was a German‑born theoretical physicist who developed a "
            "framework describing the relationship between space and "
            "time, one of the two pillars of modern physics. Einstein's mass–energy "
            "equivalence formula E = mc² has been dubbed \"the world's most famous "
            "equation.\""
        ),
    ),

    # ─────────────────────────── EXAMPLE 3 — sports  (remove “Chicago Bulls”) ───────────────────────────
    dict(
        entity="Chicago Bulls",
        original=(
            "Michael Jordan is widely regarded as the greatest basketball player "
            "of all time. Jordan won six NBA championships with the Chicago Bulls "
            "and earned five MVP awards during his career. Jordan's competitiveness "
            "and scoring prowess captivated fans worldwide."
        ),
        rewritten=(
            "Michael Jordan is widely regarded as the greatest basketball player of "
            "all time. He won six NBA championships with the franchise based in "
            "Chicago and earned five MVP awards during his career. Jordan's "
            "competitiveness and scoring prowess captivated fans worldwide."
        ),
    ),

    # ─────────────────────────── EXAMPLE 4 — technology / business  (remove “iPhone”) ───────────────────────────
    dict(
        entity="iPhone",
        original=(
            "Apple Inc. is an American multinational technology company headquartered "
            "in Cupertino, California. Apple designs, develops, and sells consumer "
            "electronics, computer software, and online services. The Apple iPhone "
            "revolutionized the smartphone industry."
        ),
        rewritten=(
            "This American multinational technology company is headquartered in "
            "Cupertino, California. It designs, develops, and sells consumer "
            "electronics, computer software, and online services. Its "
            "smartphone revolutionized the industry."
        ),
    ),

    # ─────────────────────────── EXAMPLE 5 — geography / exploration  (remove “Himalayas”) ───────────────────────────
    dict(
        entity="Himalayas",
        original=(
            "Mount Everest, located in the Himalayas, is the highest mountain on Earth "
            "with a peak at 8,848 meters above sea level. Everest attracts climbers "
            "from around the globe, including highly experienced mountaineers. The "
            "death zone on Everest poses extreme risks."
        ),
        rewritten=(
            "Mount Everest, located in a towering Asian mountain range, is the highest "
            "peak on Earth at 8,848 meters above sea level. Everest attracts "
            "climbers from around the globe, including highly experienced mountaineers. "
            "The death zone on Everest poses extreme risks."
        ),
    ),

    # ─────────────────────────── EXAMPLE 6 — history  (remove “Napoleon Bonaparte”) ───────────────────────────
    dict(
        entity="Napoleon Bonaparte",
        original=(
            "The French Revolution was a period of radical social and political change "
            "in France from 1789 to 1799. The Revolution led to the rise of Napoleon "
            "Bonaparte and profoundly affected modern history, marking the decline of "
            "monarchies. During the French Revolution, the Reign of Terror saw mass "
            "executions."
        ),
        rewritten=(
            "The French Revolution was a period of radical social and political change "
            "in France from 1789 to 1799. The upheaval paved the way for a future "
            "emperor and profoundly affected modern history, marking the decline of "
            "monarchies. During the French Revolution, the Reign of Terror saw mass "
            "executions."
        ),
    ),

    # ─────────────────────────── EXAMPLE 7 — music / pop culture  (remove “Destiny's Child”) ───────────────────────────
    dict(
        entity="Destiny's Child",
        original=(
            "Beyoncé is an American singer, songwriter, and actress who rose to fame as "
            "the lead singer of Destiny's Child. Beyoncé's solo career has won her "
            "numerous Grammy Awards and made her one of the world's best‑selling "
            "recording artists. Beyoncé also starred in major film productions."
        ),
        rewritten=(
            "Beyoncé is an American singer, songwriter, and actress who rose to fame as "
            "the lead singer of an R&B group. Beyoncé's solo career has won her "
            "numerous Grammy Awards and made her one of the world's best‑selling "
            "recording artists. Beyoncé has also starred in major film productions."
        ),
    ),
]
