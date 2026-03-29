from pipeline import CoursePlanningPipeline
import json

pipe = CoursePlanningPipeline()

print('\n' + '='*80)
print('TRANSCRIPT 1: Correct Eligibility Decision')
print('='*80)
q1 = 'Can I take CS301 Algorithms if I have completed CS201 and CS202?'
res1 = pipe.process_query(q1)

print('\n' + '='*80)
print('TRANSCRIPT 2: Course Plan Output')
print('='*80)
q2 = 'I am a BSc Computer Science student planning for Fall. I have completed CS101, CS102, MATH120, and CS201. According to the catalog, what should my course plan be? Max 3 courses.'
res2 = pipe.process_query(q2)

print('\n' + '='*80)
print('TRANSCRIPT 3: Abstention')
print('='*80)
q3 = 'When do the Spring 2025 classes start and what are the specific tuition fees for CS301?'
res3 = pipe.process_query(q3)

data = {
    't1': res1['final_output'],
    't2': res2['final_output'],
    't3': res3['final_output']
}

with open('transcripts_report.json', 'w') as f:
    json.dump(data, f, indent=2)

print("Done generating transcripts.")
