
spider_question_masking_fewshot = f"""\
<example1>
### Question: List the name, born state and age of the heads of departments ordered by age.
### Masked Question: List the [COLUMN], [COLUMN] and [COLUMN] of the [TABLE] of departments ordered by [COLUMN].
</example1>

<example2>
### Question: How many students have had at least one "B" grade?
### Masked Question: How many students have had at least one [VALUE] grade?
</example2>

<example3>
### Question: Give me the times and numbers of all trains that go to Chennai, ordered by time.
### Masked Question: Give me the [COLUMN] and [COLUMN] of all [TABLE] that go to [VALUE], ordered by [COLUMN].
</example3>
"""
