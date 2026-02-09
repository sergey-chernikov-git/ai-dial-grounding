import unittest

from task.t1.no_grounding import join_context


class TestUFunctions(unittest.TestCase):

    def test_join_context(self):
        data = [
            {
                "name": "test_name",
                "surname": "test_surname",
                "email": "test_email",
                "card": {
                    "number": "test_card_number",
                    "expiry": "test_card_expiry",
                    "cvv": "test_card_cvv"
                }
            }
        ]
        expected = "User:\n  name: test_name\n  surname: test_surname\n  email: test_email\n  card:\n    number: test_card_number\n    expiry: test_card_expiry\n    cvv: test_card_cvv\n\n"
        actual = join_context(data)
        self.assertEqual(expected, actual)
