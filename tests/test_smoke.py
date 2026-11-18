"""Run after executing notebook and exporting model artifacts."""
import os
import unittest
from pathlib import Path
from src.vinh_agents import VinhAgent

class AgentSmokeTest(unittest.TestCase):
    def test_skill_agent(self):
        path = Path(os.environ.get('VINH_MODEL_DIR', 'outputs')) / 'vietjobs_skill_extractor.joblib'
        if not path.exists():
            self.skipTest(f'No trained artifact: {path}')
        agent=VinhAgent()
        result=agent.extract_skills('Yêu cầu Python, SQL và Docker',top_k=10)
        self.assertIsInstance(result,list)
        for x in result:
            self.assertIn('skill_id', x)
            self.assertIn('skill_name', x)
            self.assertIn('confidence', x)

if __name__=='__main__': unittest.main()
