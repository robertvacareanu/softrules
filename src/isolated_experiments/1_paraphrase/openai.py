"""
Use one of the Open AI models to generate paraphrasings
"""

from string import Template

template = Template(
"""
Please generate $HOW_MANY paraphrases for the following sentence. Please ensure the meaning and the message stays the same and these two entities are preserved in your generations: "$ENTITY1", "$ENTITY2". Please be concise.
```
$TEXT
```
"""
)

