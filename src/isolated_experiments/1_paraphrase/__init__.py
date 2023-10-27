"""
Use a paraphrasing model to generate more sentences. For example:
Initial sentence: `John Doe was born in New York City`
Paraphrased sentence: `John Doe's place of birth is New York City`.

The motivation behind this experiment is the following. Let's assume we generated a rule for the initial sentence, which
looks like this: `[ne=per]+ <nsubj born >nmod_in [ne=loc]+`
We might want to train the neural matcher to be able to "match" this rule on more sentences. The question is, from where to
get those sentences. There are some immediate possible ideas:
- Rules with same relation
    + Closely related with the task we explore; should help the downstream performance
    - Potentially too ambitious. Some sentences might be extremely different, and it might be foolish to expect a good neural soft matcher to assign a high score in those cases
- Sentences where the same two entities appear (distant supervision style)
    + Unlocks a large quantity of data
    - No guarantee that the two entities are engaged in the same type of relation (e.g. `Joho Doe died in New York City`)
- Generate paraphrases for sentences already available
    + Can 2-5x the initial amount of training data (if not more, as complex sentences can allow for more paraphrases than simple sentences. 
    See an example of paraphrases for a simple sentence from ChatGPT (231027) below)
    ---------
    User 
    Please generate 5 paraphrases for the following sentence. Please ensure the meaning and the message stays the same and these two entities are preserved in your generations: "John Doe", "New York City". Please be concise.
    ```
    John Doe was born in New York City.
    ```
    ChatGPT
    Certainly, here are five concise paraphrases of the provided sentence while preserving the entities "John Doe" and "New York City":

    1. John Doe's place of birth is New York City.

    2. New York City is where John Doe was born.

    3. John Doe hails from New York City.

    4. The birthplace of John Doe is New York City.

    5. Born in New York City, John Doe is.
    ---------
    + Guaranteed (to the extent of the strength of the paraphrasing model) that the resulting sentences are similar. In other words, ideally, we would want
    the rules to match the original sentence and all its paraphrasings
"""