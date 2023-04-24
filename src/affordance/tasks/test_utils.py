import unittest

from utils import *

sample = """
Input: Q: Dominic is the star discus thrower on South's varsity track and field team. In last year's regional competition, Dominic whirled the 1.8 kg discus in a circle with a radius of 1.2 m, ultimately reaching a speed of 45 m/s before launch. Determine the net force acting upon the discus in the moments before launch.
Q1: [generate python code] write down the arithmetic or algebra equations as python code, storing the answer as 'ans'
#1:
F = mv**2/r
ans = F
print(ans)
Q2: [code execute] Execute the python code in #1 and get the value of "ans"
#2:
2160
Q3: [EOQ]
Ans: 2160
""".strip()


class TestProgram(unittest.TestCase):
    expected = """Program
======
Input: Q: Dominic is the star discus thrower on South's varsity track and field team. In last year's regional competition, Dominic whirled the 1.8 kg discus in a circle with a radius of 1.2 m, ultimately reaching a speed of 45 m/s before launch. Determine the net force acting upon the discus in the moments before launch.
------
Commands:
[generate python code]
-  input: write down the arithmetic or algebra equations as python code, storing the answer as 'ans'
- output: ↓
F = mv**2/r
ans = F
print(ans)
---
[code execute]
-  input: Execute the python code in #1 and get the value of "ans"
- output: 2160
---
[EOQ]
-  input: None
- output: None
------
Answer: 2160"""

    def test_from_to(self):
        p = Program.from_str(sample)

        self.assertEqual(self.expected, str(p))


class TestGetAnswer(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(
            get_answer(sample),
            "2160",
        )

    def test_last(self):
        self.assertEqual(
            get_answer("\n".join(sample.split("\n")[:-2])),
            "2160",
        )


if __name__ == "__main__":
    unittest.main(module="test_utils")
