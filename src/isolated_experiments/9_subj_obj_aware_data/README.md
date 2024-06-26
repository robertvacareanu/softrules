Add to the rules where both entities have the same type the SUBJECT/OBJECT position.
Add to the sentences where both entities have the same type the SUBJECT/OBJECT in their types.
Add the lexicalized entities.


`enhanced_syntax_with_lexicalized.jsonl` -> enhanced syntax rules with every rule lexicalized
`enhanced_syntax_with_lexicalized_partial.jsonl` -> enhanced syntax rules with only rules with same entity types lexicalized (rest are the same)
`enhanced_syntax_with_lexicalized_partial2.jsonl` -> enhanced syntax rules where, on top of the existing rules, we add lexicalized rules for those where both entities have the same type; So difference between `enhanced_syntax_with_lexicalized_partial2.jsonl` and `enhanced_syntax_with_lexicalized_partial.jsonl` is that `enhanced_syntax_with_lexicalized_partial2.jsonl` adds lexicalized rules on top of the existing rules; `enhanced_syntax_with_lexicalized_partial.jsonl` replaces the rules where both entities have the same type with lexicalized rules.