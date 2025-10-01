
advising_question_masking_fewshot = f"""\
<example1>
### Question: Were there any upper-level CS courses taught by instructor Seth Pettie?
### Masked Question: Were there any [VALUE] courses taught by [TABLE] [VALUE]?
</example1>

<example2>
### Question: When was the semester in which EECS 183 was offered after Winter 2016?
### Masked Question: When was the [TABLE] in which [VALUE] [VALUE] was offered after [VALUE] [VALUE]?
</example2>

<example3>
### Question: Is a lab part of the EECS 498 course?
### Masked Question: Is a [COLUMN] part of the [VALUE] [VALUE] [TABLE]?
</example3>"""

atis_question_masking_fewshot = f"""\
<example1>
### Question: Can you retrieve flights and fare IDs from BALTIMORE to PHILADELPHIA?
### Masked Question: Can you retrieve [COLUMN] and [COLUMN] from [VALUE] to [VALUE]?
</example1>

<example2>
### Question: Can you give me information on the earliest flight scheduled from NASHVILLE to TACOMA in the MORNING?
### Masked Question: Can you give me information on the earliest [TABLE] scheduled from [VALUE] to [VALUE] in the [VALUE]?
</example2>

<example3>
### Question: Give me the cities to which NW offers services.
### Masked Question: Give me the [TABLE] to which [VALUE] offers [TABLE].
</example3>"""

ehrsql_question_masking_fewshot = f"""\
<example1>
### Question: When did last patient 10014729 have the minimum value of albumin, pleural in 03/2100?
### Masked Question: When did last [TABLE] [VALUE] have the minimum value of [VALUE] in [VALUE]?
</example1>

<example2>
### Question: What are the four most frequently taken specimens for patients diagnosed with iatrogenic pneumothorax previously in the same month, in 2100?
### Masked Question: What are the [VALUE] most frequently taken [COLUMN] for [TABLE] diagnosed with [VALUE] previously in the same month, in [VALUE]?
</example2>

<example3>
### Question: Was the hematocrit value of patient 10036156 last measured on the last hospital visit less than the value second to last measured on the last hospital visit?
### Masked Question: Was the [VALUE] value of [TABLE] [VALUE] last measured on the last hospital visit less than the value [VALUE] to last measured on the last hospital visit?
</example3>"""