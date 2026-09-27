"""Synthetic-only cross-implementation checks; never opens any real run."""
import importlib.util
from pathlib import Path
import unittest

PROJECT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
independent=module('independent_audit',PROJECT/'experiments/check_dude_results.py')
production=module('production_analyzer_synthetic',PROJECT/'experiments/analyze_dude_replication.py')

def fixture(pattern):
    rows=[];records={}
    for i,correct_by_action in enumerate(pattern):
        row={'example_id':f'test-{i}','source_cluster_id':f'report-{i}',
             'original_answers':['$19.5'],'original_answer_variants':['$19.50'],
             'validated_primary_answers':['$19.5','$19.50'],'answer':'$19.5'}
        rows.append(row)
        for j,a in enumerate(independent.ACTIONS):
            response='ANSWER: '+('$19.50' if correct_by_action[j] else '$22.3')
            record={'example_id':row['example_id'],'source_cluster_id':row['source_cluster_id'],'action':a,'status':'ok',
                    'response':response,'generation_truncated':False,'elapsed_s':1.+i/7+j/10,
                    'processor_observer_elapsed_s':.01,'peak_memory_gib':12-j,'peak_reserved_gib':14,
                    'input_tokens':200+j,'visual_tokens':64+j,'generated_tokens':3+j,
                    **independent.rescore(response,row)}
            records[row['example_id'],a]=record
    return rows,records

class IndependentAuditTests(unittest.TestCase):
    def test_fully_independent_hand_calculated_self_test(self):
        self.assertEqual(independent.self_test()['status'],'PASS')

    def compare_pattern(self,pattern):
        rows,records=fixture(pattern)
        ours,_=independent.compute(rows,records)
        theirs=production.summarize(rows,records)
        self.assertEqual(independent.compare(ours,theirs),[])
        return ours

    def test_all_numeric_paths_match_on_heterogeneous_paired_fixture(self):
        pattern=[(i%3==0,i%5!=0,i%7<4,i%2==0) for i in range(66)]
        self.compare_pattern(pattern)

    def test_positive_zero_and_negative_cases_keep_gate_and_direction(self):
        positive=self.compare_pattern([(0,1,0,1)]*24+[(1,1,1,1)]*12)
        self.assertTrue(positive['quantitative_advancement']['quantitative_rule_met'])
        negative=self.compare_pattern([(0,0,1,1)]*24+[(1,1,1,1)]*12)
        self.assertFalse(negative['quantitative_advancement']['quantitative_rule_met'])
        self.assertFalse(negative['primary_result']['exact_mcnemar']['positive_direction'])
        zero=self.compare_pattern([(1,1,1,1)]*12)
        self.assertEqual(zero['primary_result']['ci95'],[0.,0.])
        self.assertFalse(zero['quantitative_advancement']['quantitative_rule_met'])

    def test_parser_invalid_and_truncated_are_retained(self):
        rows,records=fixture([(1,1,0,1)]*12)
        record=records['test-0','native_256'];record.update(response='ANSWER:',generation_truncated=True,generated_tokens=64)
        record.update(independent.rescore(record['response'],rows[0]))
        ours,_=independent.compute(rows,records);theirs=production.summarize(rows,records)
        self.assertEqual(independent.compare(ours,theirs),[])
        self.assertEqual(ours['n_records'],48)
        self.assertEqual(ours['actions']['native_256']['invalid_answers'],1)

    def test_binomial_recursion_against_exact_integer_coefficients(self):
        from math import comb
        for total in range(0,661,11):
            for count in {0,total//7,total//2,total}:
                expected=min(1.,2*sum(comb(total,k) for k in range(min(count,total-count)+1))/(1<<total))
                self.assertAlmostEqual(independent.exact_binomial(count,total-count),expected,places=13)

    def test_unapproved_variants_and_duplicate_source_fail(self):
        rows,records=fixture([(1,1,0,1)]*2)
        rows[0]['validated_primary_answers'].append('invented')
        with self.assertRaises(ValueError):independent.compute(rows,records)
        rows[0]['validated_primary_answers'].pop();rows[1]['source_cluster_id']=rows[0]['source_cluster_id']
        with self.assertRaises(ValueError):independent.compute(rows,records)

if __name__=='__main__':unittest.main(verbosity=2)
