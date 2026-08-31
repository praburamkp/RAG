import unittest

from banking_tower import InvestmentBankingTower


class InvestmentBankingTowerTests(unittest.TestCase):
    def test_routes_valuation_request_and_returns_auditable_task(self):
        tower = InvestmentBankingTower()
        task = tower.send_message("Prepare a DCF valuation and assess debt leverage.")

        self.assertEqual(task.status, "completed")
        self.assertTrue(task.artifacts)
        report = task.artifacts[0].text
        self.assertIn("Valuation", report)
        self.assertIn("Capital structure", report)
        self.assertIn("Risk and compliance", report)

    def test_card_publishes_specialist_skills(self):
        card = InvestmentBankingTower().agent_card().to_dict()
        self.assertEqual(card["protocolVersion"], "0.3")
        self.assertEqual(len(card["skills"]), 5)


if __name__ == "__main__":
    unittest.main()
