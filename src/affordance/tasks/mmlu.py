import argparse
from collections import Counter
import csv
import os
import pdb
import re

import numpy as np
from prompt_library import (
    few_shot_retrieval_prompt,
    llm_similar_tasks,
    random_tasks,
    similar_auto_breakdowns,
    similar_tasks,
)
from sequential_interpreter import TopDownVisitorBeta
from tqdm import tqdm
from transformers import GPT2Tokenizer
from utils import (
    OpenAIModel,
    cache_dir,
    chunks,
    get_answer,
    get_autocot_answer,
    get_few_shot_prompt,
    substring_match,
)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

few_shot_cot_prompt = few_shot_retrieval_prompt

# task_description = "MMLU dataset"

# auto_cot_cleaned_prompt = """Answer the following multiple-choice questions about college-level Computer Science, which tests for specialized knowledge and concepts like Algorithms, systems, graphs, recursion, etc.
# Which of the following regular expressions is equivalent to (describes the same set of strings as) (a* + b)*(c + d)?
# A) a*(c + d)+ b(c + d)
# B) a*(c + d)* + b(c + d)*
# C) a*(c + d)+ b*(c + d)
# D) (a + b)*c +(a + b)*d
# A: Let's think step-by-step.

# The first part of the expression, (a* + b)*, describes any string that is a combination of zero or more a's, followed by zero or more b's.
# The second part of the expression, (c + d), describes any string that is either c or d.
# Putting these together, the entire expression describes any string that is a combination of zero or more a's, followed by zero or more b's, followed by either c or d.
# This is exactly the same set of strings as described by (a* + b)*(c + d). Therefore, the answer is C.

# The final answer is C
# ----
# Answer the following multiple-choice questions about college-level Computer Science, which tests for specialized knowledge and concepts like Algorithms, systems, graphs, recursion, etc.
# A certain pipelined RISC machine has 8 general-purpose registers R0, R1, . . . , R7 and supports the following operations.
# ADD Rs1, Rs2, Rd Add Rs1 to Rs2 and put the sum in Rd
# MUL Rs1, Rs2, Rd Multiply Rs1 by Rs2 and put the product in Rd
# An operation normally takes one cycle; however, an operation takes two cycles if it produces a result required by the immediately following operation in an operation sequence. Consider the expression AB + ABC + BC, where variables A, B, C are located in registers R0, R1, R2. If the contents of these three registers must not be modified, what is the minimum number of clock cycles required for an operation sequence that computes the value of AB + ABC + BC?
# A) 5
# B) 6
# C) 7
# D) 8
# A: Let's think step-by-step.

# The first operation is AB. This will take 1 cycle.
# The second operation is ABC. This will take 2 cycles because the result of AB is required for this operation.
# The third operation is BC. This will take 1 cycle.
# So in total, it will take 5 cycles.

# The final answer is A
# ----
# Answer the following multiple-choice questions about college-level Computer Science, which tests for specialized knowledge and concepts like Algorithms, systems, graphs, recursion, etc.
# The Singleton design pattern is used to guarantee that only a single instance of a class may be instantiated. Which of the following is (are) true of this design pattern?
# I. The Singleton class has a static factory method to provide its instance.
# II. The Singleton class can be a subclass of another class.
# III. The Singleton class has a private constructor.
# A) I only
# B) II only
# C) III only
# D) I, II, and III
# A: Let's think step-by-step.

# The Singleton design pattern is used to guarantee that only a single instance of a class may be instantiated.
# This means that we need to have some way of ensuring that only one instance of the class can be created. One way to do this is to have a static factory method that provides the instance. Another way to do this is to have a private constructor so that no other class can instantiate the Singleton class.
# So, the answer is D) I, II, and III.

# The final answer is D
# ----
# Answer the following multiple-choice questions about college-level Computer Science, which tests for specialized knowledge and concepts like Algorithms, systems, graphs, recursion, etc.
# A compiler generates code for the following assignment statement.
# G := (A + B) * C - (D + E) * F
# The target machine has a single accumulator and a single-address instruction set consisting of instructions load, store, add, subtract, and multiply. For the arithmetic operations, the left operand is taken from the accumulator and the result appears in the accumulator. The smallest possible number of instructions in the resulting code is
# A) 5
# B) 6
# C) 7
# D) 9
# A: Let's think step-by-step.

# The first thing we need to do is compute (A + B). We can do this in two steps: first, we load A into the accumulator, then we add B to it.
# Next, we need to compute C * (A + B). We can do this in two steps as well: first, we load C into the accumulator, then we multiply it by (A + B).
# Now we need to compute (D + E). We can do this in two steps: first, we load D into the accumulator, then we add E to it.
# Finally, we need to compute F * (D + E). We can do this in two steps: first, we load F into the accumulator, then we multiply it by (D + E).
# The overall result is that we need 7 instructions to compute the expression G := (A + B) * C - (D + E) * F.

# The final answer is C
# ----
# Answer the following multiple-choice questions about college-level Computer Science, which tests for specialized knowledge and concepts like Algorithms, systems, graphs, recursion, etc.
# Consider a computer design in which multiple processors, each with a private cache memory, share global memory using a single bus. This bus is the critical system resource. Each processor can execute one instruction every 500 nanoseconds as long as memory references are satisfied by its local cache. When a cache miss occurs, the processor is delayed for an additional 2,000 nanoseconds. During half of this additional delay, the bus is dedicated to serving the cache miss. During the other half, the processor cannot continue, but the bus is free to service requests from other processors. On average, each instruction requires 2 memory references. On average, cache misses occur on 1 percent of references. What proportion of the capacity of the bus would a single processor consume, ignoring delays due to competition from other processors?
# A) 1/50
# B) 1/27
# C) 1/25
# D) 2/27
# A: Let's think step-by-step.

# We're given that each instruction requires 2 memory references, on average. We're also given that cache misses occur on 1% of references, on average.
# This means that, on average, each instruction will result in 1 cache miss.
# We're told that when a cache miss occurs, the processor is delayed for an additional 2,000 nanoseconds. During half of this additional delay, the bus is dedicated to serving the cache miss.
# This means that, on average, each cache miss will result in the processor being delayed for an additional 1,000 nanoseconds.
# We're told that each processor can execute one instruction every 500 nanoseconds as long as memory references are satisfied by its local cache.
# This means that, on average, each processor can execute 2 instructions per second.
# We're asked to find the proportion of the capacity of the bus that a single processor would consume.
# This means we need to find the average number of instructions that a single processor can execute per second, taking into account the delays due to cache misses.
# We can do this by multiplying the average number of instructions that a single processor can execute per second (2) by the average number of cache misses per instruction (1%).
# This gives us an average of 0.02 instructions per second that a single processor can execute, taking into account the delays due to cache misses.
# This means that a single processor would consume 1/50 of the capacity of the bus.

# The final answer is A
# ----
# """
# dataset_name = "college_computer_science"
# task_name = "College computer science"
# task_description = "Answer the following multiple-choice questions about college-level Computer Science, which tests for specialized knowledge and concepts like Algorithms, systems, graphs, recursion, etc."

# auto_cot_cleaned_prompt = """Answer the following multiple-choice questions about college-level Astronomy, which tests for specialized knowledge and concepts like ASolar system, galaxies, asteroids, etc.
# You are pushing a truck along a road. Would it be easier to accelerate this truck on Mars? Why? (Assume there is no friction)
# A) It would be harder since the truck is heavier on Mars.
# B) It would be easier since the truck is lighter on Mars.
# C) It would be harder since the truck is lighter on Mars.
# D) It would be the same no matter where you are.
# A: Let's think step-by-step.

# First, we need to understand what it means to "accelerate." In physics, acceleration is the rate of change of velocity. So, if we want to accelerate the truck, we need to increase its velocity.
# Now, let's think about what factors would affect the truck's velocity. One factor is the truck's mass. If the truck has more mass, it will be harder to accelerate because it takes more force to move a more massive object. Another factor is gravity. If the truck is on a planet with more gravity, it will be harder to accelerate because the force of gravity will be pulling the truck down.
# So, on Mars, the truck would have less mass than on Earth, but it would also be subject to more gravity. These two factors would cancel each other out, and the truck would accelerate at the same rate no matter where it is.

# The final answer is D
# ----
# Answer the following multiple-choice questions about college-level Astronomy, which tests for specialized knowledge and concepts like ASolar system, galaxies, asteroids, etc.
# Where do most short-period comets come from and how do we know?
# A) The Kuiper belt; short period comets tend to be in the plane of the solar system just like the Kuiper belt.
# B) The Kuiper belt; short period comets tend to come from random directions indicating a spherical distribution of comets called the Kuiper belt.
# C) The asteroid belt; short period comets have orbital periods similar to asteroids like Vesta and are found in the plane of the solar system just like the asteroid belt.
# D) The Oort cloud; short period comets tend to be in the plane of the solar system just like the Oort cloud.
# A: Let's think step-by-step.

# First, we know that most short-period comets come from the Kuiper belt. This is because the Kuiper belt is a spherical distribution of comets, and short period comets tend to be in the plane of the solar system.
# Second, we know that the Kuiper belt is the source of most short-period comets because short period comets have orbital periods similar to asteroids like Vesta. This means that they are found in the plane of the solar system, just like the asteroid belt.

# The final answer is A
# ----
# Answer the following multiple-choice questions about college-level Astronomy, which tests for specialized knowledge and concepts like ASolar system, galaxies, asteroids, etc.
# Say the pupil of your eye has a diameter of 5 mm and you have a telescope with an aperture of 50 cm. How much more light can the telescope gather than your eye?
# A) 10000 times more
# B) 100 times more
# C) 1000 times more
# D) 10 times more
# A: Let's think step-by-step.

# The area of the pupil of the eye is:

# A = πr2
# A = π(5 mm)2
# A = 78.54 mm2

# The area of the telescope's aperture is:

# A = πr2
# A = π(50 cm)2
# A = 1963.5 cm2

# The ratio of the two areas is:

# Aeye/Atelescope = 78.54 mm2/1963.5 cm2
# Aeye/Atelescope = 0.04 mm2/cm2

# Now we can find out how much more light the telescope can gather than the eye. We just need to take the inverse of the ratio of the two areas:

# Atelescope/Aeye = 1/(Aeye/Atelescope)
# Atelescope/Aeye = 1/(0.04 mm2/cm2)
# Atelescope/Aeye = 25 cm2/mm2

# This means that the telescope can gather 25 times more light than the eye.

# The final answer is 10.
# ----
# Answer the following multiple-choice questions about college-level Astronomy, which tests for specialized knowledge and concepts like ASolar system, galaxies, asteroids, etc.
# Why isn't there a planet where the asteroid belt is located?
# A) A planet once formed here but it was broken apart by a catastrophic collision.
# B) There was not enough material in this part of the solar nebula to form a planet.
# C) There was too much rocky material to form a terrestrial planet but not enough gaseous material to form a jovian planet.
# D) Resonance with Jupiter prevented material from collecting together to form a planet.
# A: Let's think step-by-step.

# The asteroid belt is located between Mars and Jupiter.
# A planet once formed here but it was broken apart by a catastrophic collision - This could be a possibility, but there is no evidence to support this.
# There was not enough material in this part of the solar nebula to form a planet - This is a possibility. The asteroid belt is made up of left over material from the formation of the solar system.
# There was too much rocky material to form a terrestrial planet but not enough gaseous material to form a jovian planet - This is a possibility. The asteroid belt is made up of rocky material, which would be too heavy to form a gas planet.
# Resonance with Jupiter prevented material from collecting together to form a planet - This is a possibility. The asteroid belt is in a 3:2 resonance with Jupiter, which means that for every 3 orbits Jupiter completes, the asteroid belt completes 2. This could have prevented material from coming together to form a planet.

# The final answer is B D or D
# ----
# Answer the following multiple-choice questions about college-level Astronomy, which tests for specialized knowledge and concepts like ASolar system, galaxies, asteroids, etc.
# Why is Mars red?
# A) Because the surface is covered with heavily oxidized ("rusted") minerals.
# B) Because the atmosphere scatters more light at bluer wavelengths transmitting mostly red light.
# C) Because Mars is covered with ancient lava flows which are red in color.
# D) Because flowing water on Mars's surface altered the surface minerals several billion years ago.
# A: Let's think step-by-step.

# The surface of Mars is red because it is covered with heavily oxidized ("rusted") minerals.

# The final answer is A
# ----
# """
# dataset_name = "astronomy"
# task_name = "Astronomy"
# task_description = "Answer the following multiple-choice questions about college-level Astronomy, which tests for specialized knowledge and concepts like ASolar system, galaxies, asteroids, etc."

# auto_cot_cleaned_prompt = """Answer the following multiple-choice questions about college-level Business Ethics, which tests for specialized knowledge and concepts like Corporate responsibility, stakeholders, regulation, etc.
# Beyond the business case for engaging in CSR there are a number of moral arguments relating to: negative _______, the _______that corporations possess and the ________ of business and society.
# A) Externalities, Power, Independence
# B) Publicity, Insubstantial resources, Mutual dependence
# C) Externalities, Power, Mutual dependence
# D) Publicity, Power, Independence
# A: Let's think step-by-step.

# The first part of the question asks about the "business case for engaging in CSR." This is referring to the idea that there are benefits to a company engaging in CSR activities, even beyond the simple act of being morally good.
# The second part of the question asks about "moral arguments relating to negative externalities, the power that corporations possess, and the independence of business and society."
# Negative externalities are when a company's activities have a negative impact on society, even if that was not the company's intention. For example, if a company pollutes a river, that would be a negative externality.
# The power that corporations possess refers to the fact that corporations are large, often multinational organizations with a lot of money and resources. This can give them a lot of power in society.
# The independence of business and society means that businesses should not be completely independent from society, but rather should be working together for the common good.
# So, based on this information, the correct answer is D: Externalities, Power, Mutual dependence.

# The final answer is C
# ----
# Answer the following multiple-choice questions about college-level Business Ethics, which tests for specialized knowledge and concepts like Corporate responsibility, stakeholders, regulation, etc.
# _______ is the direct attempt to formally or informally manage ethical issues or problems, through specific policies, practices and programmes.
# A) Corporate social responsibility
# B) Business ethics management
# C) Sustainability
# D) Environmental management
# A: Let's think step-by-step.

# The first step is to identify the key concepts in the question stem. In this case, they are "formally or informally manage ethical issues or problems," "specific policies, practices and programmes," and "corporate social responsibility."
# The second step is to look at the answer choices and see which one best matches the key concepts. In this case, the answer is B) Business ethics management.

# The final answer is B
# ----
# Answer the following multiple-choice questions about college-level Business Ethics, which tests for specialized knowledge and concepts like Corporate responsibility, stakeholders, regulation, etc.
# To ensure the independence of the non-executive board members, they are a number of steps which can be taken, which include non-executives being drawn from _______ the company, being appointed for a _________ time period as well as being appointed _________.
# A) Outside, Limited, Independently
# B) Inside, Limited, Intermittently
# C) Outside, Unlimited, Intermittently
# D) Inside, Unlimited, Independently
# A: Let's think step-by-step.

# To ensure the independence of the non-executive board members:

# 1) They can be drawn from outside the company
# 2) They can be appointed for a limited time period
# 3) They can be appointed independently

# The final answer is A
# ----
# Answer the following multiple-choice questions about college-level Business Ethics, which tests for specialized knowledge and concepts like Corporate responsibility, stakeholders, regulation, etc.
# Three contrasting tactics that CSO's can engage in to meet their aims are ________ which typically involves research and communication, ________, which may involve physically attacking a company's operations or ________, often involving some form of _______.
# A) Non-violent direct action, Violent direct action, Indirect action, Boycott
# B) Indirect action, Instrumental action, Non-violent direct action, Information campaign
# C) Indirect action, Violent direct action, Non-violent direct-action Boycott
# D) Non-violent direct action, Instrumental action, Indirect action, Information campaign
# A: Let's think step-by-step.

# The first step is to identify the three contrasting tactics that CSO's can engage in. These are:

# 1) Non-violent direct action
# 2) Violent direct action
# 3) Indirect action

# The next step is to determine which of these tactics typically involves research and communication. This would be non-violent direct action.
# The next step is to determine which of these tactics may involve physically attacking a company's operations. This would be violent direct action.
# The next step is to determine which of these tactics often involves some form of boycott. This would be indirect action.

# The final answer is A
# ----
# Answer the following multiple-choice questions about college-level Business Ethics, which tests for specialized knowledge and concepts like Corporate responsibility, stakeholders, regulation, etc.
# In contrast to _______, _______ aim to reward favourable behaviour by companies. The success of such campaigns have been heightened through the use of ___________, which allow campaigns to facilitate the company in achieving _________ .
# A) Buycotts, Boycotts, Blockchain technology, Charitable donations
# B) Buycotts, Boycotts, Digital technology, Increased Sales
# C) Boycotts, Buyalls, Blockchain technology, Charitable donations
# D) Boycotts, Buycotts, Digital technology, Increased Sales
# A: Let's think step-by-step.

# The first word is "In contrast to." So, we are looking for two words that are opposites.
# The second word is "aim." So, we are looking for two words that have different purposes.
# The third word is "have been heightened." So, we are looking for two words that describe different levels of success.
# The fourth word is "allow." So, we are looking for two words that describe different things that campaigns can do.
# From this, we can eliminate choices B and C.
# Choice A is not correct because buycotts and boycotts are not opposites.
# Choice D is correct because boycotts and buycotts have different purposes (one is to punish, the other is to reward), and the use of digital technology has increased the success of such campaigns.

# The final answer is D
# ----
# """
# dataset_name = "business_ethics"
# task_name = "Business Ethics"
# task_description = "Answer the following multiple-choice questions about college-level Business Ethics, which tests for specialized knowledge and concepts like Corporate responsibility, stakeholders, regulation, etc."

auto_cot_cleaned_prompt = """Answer the following multiple-choice questions about college-level Virology, which tests for specialized knowledge and concepts like Epidemiology, coronaviruses, retroviruses, herpesviruses, etc.
Why are parvoviruses a highly impactful parasite?
A) Because they have no nucleic acid
B) They require a helper virus
C) Only replicate in dividing cells
D) Can integrate into host chromosomes
A: Let's think step-by-step.

First, what are viruses? They are tiny infectious particles that can replicate only inside the living cells of an organism.

Second, what are the three main types of viruses?
-Bacteria-infecting viruses (bacteriophages)
-Plant-infecting viruses
-Animal-infecting viruses

Third, what are the four main types of animal-infecting viruses?
-Retroviruses
-Herpesviruses
-Coronaviruses
-Parvoviruses

Fourth, what is the difference between a retrovirus and a parvovirus?
-Retroviruses are RNA viruses that use reverse transcriptase to convert their RNA into DNA. This DNA then integrates into the host cell's genome.
-Parvoviruses are DNA viruses that only replicate in dividing cells. They do not integrate into the host cell's genome.

So, why are parvoviruses a highly impactful parasite?
The answer is C: They only replicate in dividing cells.
Since parvoviruses only replicate in dividing cells, they can easily infect and kill cells that are rapidly dividing, such as cancer cells. This makes them a highly impactful parasite.

The final answer is C
----
Answer the following multiple-choice questions about college-level Virology, which tests for specialized knowledge and concepts like Epidemiology, coronaviruses, retroviruses, herpesviruses, etc.
Which of the following is a morphological characteristic of the paramyxoviruses.
A) Fragile viruses often visualised with RNA spewing from the inside
B) Elongate viruses
C) Icosahedral viruses with envelope
D) Very large viruses
A: Let's think step-by-step.

First, what are the morphological characteristics of paramyxoviruses?

According to the Centers for Disease Control and Prevention (CDC), paramyxoviruses are "spherical or pleomorphic (abnormally shaped) viruses with a diameter of approximately 150 nanometers (nm)".
Next, let's look at each of the answer choices and see if they match the description of paramyxoviruses.

A) Fragile viruses often visualised with RNA spewing from the inside
This answer choice does not match the description of paramyxoviruses. Paramyxoviruses are not fragile, and they do not spew RNA from the inside.

B) Elongate viruses
This answer choice does not match the description of paramyxoviruses. Paramyxoviruses are not elongate.

C) Icosahedral viruses with envelope
This answer choice matches the description of paramyxoviruses. Paramyxoviruses are icosahedral, meaning they have 20 sides, and they have an envelope.

D) Very large viruses
This answer choice does not match the description of paramyxoviruses. Paramyxoviruses are not very large.

The final answer is C
----
Answer the following multiple-choice questions about college-level Virology, which tests for specialized knowledge and concepts like Epidemiology, coronaviruses, retroviruses, herpesviruses, etc.
A key factor facilitating the application of nested case-control studies from the MACS was:
A) Data collection
B) Establishment of a repository of biologic specimens
C) Participant interest
D) Administration of the questionnaire by staff
A: Let's think step-by-step.

First, you need a population from which you can draw your cases and controls. This is usually done by establishing a repository of biologic specimens. Then, you need to collect data on each individual in the population, including information on their exposures and outcomes. Finally, you need to administer the questionnaire to the participants.
So the answer is B) Establishment of a repository of biologic specimens.

The final answer is B
----
Answer the following multiple-choice questions about college-level Virology, which tests for specialized knowledge and concepts like Epidemiology, coronaviruses, retroviruses, herpesviruses, etc.
The most important goal of a behavioral intervention is:
A) Change in behavior
B) Comprehensive coverage
C) Effective use of behavioral theory
D) Sustained behavior change
A: Let's think step-by-step.

The most important goal of a behavioral intervention is to change behavior. This is the ultimate goal, and everything else is secondary. That being said, comprehensive coverage (of the population), effective use of behavioral theory, and sustained behavior change are all important factors in achieving the ultimate goal.

The final answer is A
----
Answer the following multiple-choice questions about college-level Virology, which tests for specialized knowledge and concepts like Epidemiology, coronaviruses, retroviruses, herpesviruses, etc.
The median survival time to AIDS and death was established by following:
A) Seroprevalent HIV-infected individuals
B) Seroconverters
C) Seronegatives
D) High-risk seronegatives
A: Let's think step-by-step.

The median survival time to AIDS and death was established by following:

A) Seroprevalent HIV-infected individuals
This means that the researchers looked at a group of people who were already infected with HIV and tracked how long they lived. The median is the middle value, so this would give them an idea of how long, on average, people with HIV live.

B) Seroconverters
This means that the researchers looked at a group of people who became infected with HIV and tracked how long they lived. This would give them an idea of how long, on average, people live after becoming infected with HIV.

C) Seronegatives
This means that the researchers looked at a group of people who did not have HIV and tracked how long they lived. This would give them an idea of how long, on average, people without HIV live.

D) High-risk seronegatives
This means that the researchers looked at a group of people who did not have HIV but were at high risk for becoming infected (for example, they may have been exposed to the virus or have a weakened immune system). This would give them an idea of how long, on average, people without HIV live when they are at high risk for infection.

The final answer is B
----
"""
dataset_name = "virology"
task_name = "Virology"
task_description = "Answer the following multiple-choice questions about college-level Virology, which tests for specialized knowledge and concepts like Epidemiology, coronaviruses, retroviruses, herpesviruses, etc."

# auto_cot_cleaned_prompt = """Answer the following multiple-choice questions about college-level High School Geography, which tests for specialized knowledge and concepts like Population migration, rural land-use, urban processes, etc.
# # The rate of natural increase of a population is found by subtracting the
# # A) crude death rate from the crude birth date.
# # B) crude birth rate from the crude death rate.
# # C) doubling time from the crude birth rate.
# # D) fertility rate from the crude death rate.
# # A: Let's think step-by-step.

# # The rate of natural increase of a population is found by subtracting the crude death rate from the crude birth rate.

# # The final answer is A
# # ----
# # Answer the following multiple-choice questions about college-level High School Geography, which tests for specialized knowledge and concepts like Population migration, rural land-use, urban processes, etc.
# # During the third stage of the demographic transition model, which of the following is true?
# # A) Birth rates increase and population growth rate is less rapid.
# # B) Birth rates decline and population growth rate is less rapid.
# # C) Birth rates increase and population growth rate increases.
# # D) Birth rates decrease and population growth rate increases.
# # A: Let's think step-by-step.

# # The third stage of the demographic transition model is when birth rates decline and the population growth rate is less rapid.

# # The final answer is B
# # ----
# # Answer the following multiple-choice questions about college-level High School Geography, which tests for specialized knowledge and concepts like Population migration, rural land-use, urban processes, etc.
# # Which of the following statements is NOT accurate regarding the services provided by local governments in the United States?
# # A) Duplication of efforts occurs often.
# # B) Social problems of the central city spill over into the surrounding residential suburbs.
# # C) Inefficiency in providing services occurs often.
# # D) One neighborhood's efforts to reduce pollution are always supported by neighboring communities.
# # A: Let's think step-by-step.

# # First, we need to identify what the question is asking. The question is asking us to identify which of the following statements is NOT accurate regarding the services provided by local governments in the United States.

# # Next, we need to read each answer choice and determine if it is accurate or not.

# # A) Duplication of efforts occurs often. - This is accurate. Local governments often duplicate efforts because they are not coordinated.
# # B) Social problems of the central city spill over into the surrounding residential suburbs. - This is accurate. Social problems in the central city often spill over into the surrounding residential suburbs.
# # C) Inefficiency in providing services occurs often. - This is accurate. Local governments often provide services inefficiently.
# # D) One neighborhood's efforts to reduce pollution are always supported by neighboring communities. - This is NOT accurate. One neighborhood's efforts to reduce pollution are not always supported by neighboring communities.

# # The final answer is D
# # ----
# # Answer the following multiple-choice questions about college-level High School Geography, which tests for specialized knowledge and concepts like Population migration, rural land-use, urban processes, etc.
# # The practice of hiring a foreign third-party service provider to run an operation is called
# # A) outsourcing.
# # B) offshoring.
# # C) maquiladoras.
# # D) locational interdependence.
# # A: Let's think step-by-step.

# # The first step is to identify the keyword in the question, which in this case is "outsourcing." Based on the definition provided, we can eliminate options B) and D).
# # The second step is to look at the remaining options and see if we can eliminate any based on what we know. In this case, we can eliminate option C) because it is not related to outsourcing.
# # This leaves us with option A) as the correct answer.

# # The final answer is A
# # ----
# # Answer the following multiple-choice questions about college-level High School Geography, which tests for specialized knowledge and concepts like Population migration, rural land-use, urban processes, etc.
# # Which one of the following items is an example of nonmaterial culture?
# # A) Dove soap
# # B) Dove candy bar
# # C) Dove symbol
# # D) A dove (bird)
# # A: Let's think step-by-step.

# # First, we need to understand what nonmaterial culture is. Nonmaterial culture refers to the nonphysical things that people believe, think, and feel. This includes things like language, religion, and values.

# # Now that we know that, we can look at the answer choices and see which one best fits the definition of nonmaterial culture.

# # A) Dove soap - This is a physical object and does not fit the definition of nonmaterial culture.
# # B) Dove candy bar - This is a physical object and does not fit the definition of nonmaterial culture.
# # C) Dove symbol - This could possibly be an example of nonmaterial culture, as it is something that people believe in or think about.
# # D) A dove (bird) - This is a physical object and does not fit the definition of nonmaterial culture.
# # The best answer choice is C) Dove symbol.

# # The final answer is C
# # ----
# # """
# dataset_name = "high_school_geography"
# task_name = "High School Geography"
# task_description = "Answer the following multiple-choice questions about college-level High School Geography, which tests for specialized knowledge and concepts like Population migration, rural land-use, urban processes, etc."

# few_shot_cot_prompt = few_shot_arithmetic_prompt
# auto_cot_cleaned_prompt = """Answer the following multiple-choice questions about college-level High School Mathematics, which tests for specialized knowledge and concepts like Pre-algebra, algebra, trigonometry, calculus, etc.
# Joe was in charge of lights for a dance. The red light blinks every two seconds, the yellow light every three seconds, and the blue light every five seconds. If we include the very beginning and very end of the dance, how many times during a seven minute dance will all the lights come on at the same time? (Assume that all three lights blink simultaneously at the very beginning of the dance.)
# A) 3
# B) 15
# C) 6
# D) 5
# A: Let's think step-by-step.

# There are 60 seconds in a minute, so there are 420 seconds in a 7-minute dance.
# The red light blinks every 2 seconds, so it will blink 210 times during the dance.
# The yellow light blinks every 3 seconds, so it will blink 140 times during the dance.
# The blue light blinks every 5 seconds, so it will blink 84 times during the dance.
# Now, we just need to find how many times 210, 140, and 84 will all be equal to each other. We can do this by finding the greatest common factor of 210, 140, and 84.
# The greatest common factor of 210, 140, and 84 is 2 * 3 * 5 = 30.
# This means that all three lights will be on together 30 times during the dance.
# Answer: C) 6

# The final answer is C
# ----
# Answer the following multiple-choice questions about college-level High School Mathematics, which tests for specialized knowledge and concepts like Pre-algebra, algebra, trigonometry, calculus, etc.
# Five thousand dollars compounded annually at an $x\%$ interest rate takes six years to double. At the same interest rate, how many years will it take $\$300$ to grow to $\$9600$?
# A) 12
# B) 1
# C) 30
# D) 5
# A: Let's think step-by-step.

# We know that $5000 \times (1 + x\%)^6 = 10000$.

# We also know that $300 \times (1 + x\%)^n = 9600$.

# We can set these two equations equal to each other and solve for $n$:

# $5000 \times (1 + x\%)^6 = 10000$
# $300 \times (1 + x\%)^n = 9600$

# $(1 + x\%)^6 = \frac{10000}{5000}$
# $(1 + x\%)^n = \frac{9600}{300}$

# $(1 + x\%)^6 = 2$
# $(1 + x\%)^n = \frac{32}{5}$

# $6 = n$

# Therefore, it will take 6 years for $\$300$ to grow to $\$9600$ at the same interest rate.

# The final answer is D
# ----
# Answer the following multiple-choice questions about college-level High School Mathematics, which tests for specialized knowledge and concepts like Pre-algebra, algebra, trigonometry, calculus, etc.
# The variable $x$ varies directly as the square of $y$, and $y$ varies directly as the cube of $z$. If $x$ equals $-16$ when $z$ equals 2, what is the value of $x$ when $z$ equals $\frac{1}{2}$?
# A) -1
# B) 16
# C) -\frac{1}{256}
# D) \frac{1}{16}
# A: Let's think step-by-step.

# We know that $x$ varies directly as the square of $y$, so we can write $x = ky^2$ for some constant $k$.

# Similarly, we know that $y$ varies directly as the cube of $z$, so we can write $y = mz^3$ for some constant $m$.

# Now, we can plug these expressions for $x$ and $y$ into our original equation to get:

# $$x = ky^2 = k(mz^3)^2 = km^2z^6$$

# We know that when $z = 2$, $x = -16$, so we can plug this into our equation to get:

# $$-16 = km^2(2)^6 \Rightarrow k = \frac{-16}{m^2(2)^6}$$

# Now that we have our value for $k$, we can plug it back into our expression for $x$ to get:

# $$x = \frac{-16}{m^2(2)^6}m^2z^6 = \frac{-16}{(2)^6}z^6 = \frac{-1}{2^6}z^6$$

# Finally, we can plug in $z = \frac{1}{2}$ to get:

# $$x = \frac{-1}{2^6}\left(\frac{1}{2}\right)^6 = \frac{-1}{2^{12}} = \frac{-1}{4096}$$

# The final answer is C
# ----
# Answer the following multiple-choice questions about college-level High School Mathematics, which tests for specialized knowledge and concepts like Pre-algebra, algebra, trigonometry, calculus, etc.
# Simplify and write the result with a rational denominator: $$\sqrt{\sqrt[3]{\sqrt{\frac{1}{729}}}}$$
# A) \frac{3\sqrt{3}}{3}
# B) \frac{1}{3}
# C) \sqrt{3}
# D) \frac{\sqrt{3}}{3}
# A: Let's think step-by-step.

# $\sqrt{\sqrt[3]{\sqrt{\frac{1}{729}}}}$
# $=\sqrt[3]{\sqrt{\frac{1}{729}}}$
# $=\sqrt{\frac{1}{3}}$
# $=\frac{1}{3}$

# The final answer is B
# ----
# """
# dataset_name = "high_school_mathematics"
# task_name = "High School Mathematics"
# task_description = "Answer the following multiple-choice questions about college-level High School Mathematics, which tests for specialized knowledge and concepts like Pre-algebra, algebra, trigonometry, calculus, etc."


mmlu_namespace = os.path.join(cache_dir, "mmlu", "data")
data_files = {
    "train": os.path.join(mmlu_namespace, "dev/{0}_dev.csv".format(dataset_name)),
    "validation": os.path.join(mmlu_namespace, "test/{0}_test.csv".format(dataset_name)),
}
train_data = [row for row in csv.reader(open(data_files["train"]))]
train_inputs = [
    row[0]
    + "\n"
    + "\n".join(
        ["%s) %s" % (letter, choice) for letter, choice in zip(["A", "B", "C", "D"], row[1:-1])]
    )
    for row in train_data
]
train_labels = [row[-1] for row in train_data]

data = [row for row in csv.reader(open(data_files["validation"]))]
inputs = [
    row[0]
    + "\n"
    + "\n".join(
        ["%s) %s" % (letter, choice) for letter, choice in zip(["A", "B", "C", "D"], row[1:-1])]
    )
    for row in data
]
labels = [row[-1] for row in data]

io_pairs = [(inp, lab) for inp, lab in zip(train_inputs[:5], train_labels[:5])]
# inputs = d['train']['inputs'] + d['validation']['inputs']
# # inputs = [x.split('\n')[0] for x in inputs]
# labels = d['train']['targets'] + d['validation']['targets']
io_prompt = "\n----\n".join([(inp + "\nAns:\n" + lab) for (inp, lab) in io_pairs])


def exact_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() == predict.lower():
            correct += 1
        count += 1
    return (1.0 * correct) / count


def few_shot(N=10, temperature=0.3, model_name="text-davinci-002"):
    few_shot_prompt = get_few_shot_prompt(train_inputs, train_labels, n=N)
    print(len(tokenizer(few_shot_prompt)["input_ids"]))

    def predict(chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=200, temperature=temperature, quote="---", n=1
        )
        prompts = [
            """%s\
%s"""
            % (few_shot_prompt, x)
            for x in chunk
        ]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            answers.extend(predict(x))
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(labels, preds))
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


auto_cot_corrected_prompt = """"""


def auto_cot(
    temperature=0.3,
    model_name="text-davinci-002",
    predict=True,
    use_corrected=False,
    self_consistency=False,
):
    global auto_cot_corrected_prompt
    global auto_cot_cleaned_prompt
    auto_cot_prompt = ""
    for io_pair in io_pairs[:10]:
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompt = """%s\n""" % task_description + io_pair[0] + """\nA: Let's think step-by-step.\n"""
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"
        # Add the final answer with special format so evaluation is easier.

    if use_corrected:
        auto_cot_prompt = auto_cot_corrected_prompt
    else:
        auto_cot_prompt = auto_cot_cleaned_prompt

    if not predict:
        return

    print(auto_cot_prompt)
    f = open("auto_cot_demonstrations.txt", "a+")
    f.write("Computer Science\n\n")
    f.write(auto_cot_prompt)

    def predict_self_consistency(description, chunk, n=5):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=n
        )
        prompts = [
            auto_cot_prompt
            + """%s\n""" % task_description
            + """%s\nA: Let's think step-by-step.\n""" % (x)
            for x in chunk
        ]
        return gpt3(prompts)

    def predict(chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=500, temperature=temperature, quote="---", n=1
        )
        prompts = [
            auto_cot_prompt
            + """%s\n""" % task_description
            + """%s\nA: Let's think step-by-step.\n""" % (x)
            for x in chunk
        ]
        return gpt3(prompts)

    if self_consistency:
        perf_array = []
        runs = 1
        batch_size = 2
        for run in range(runs):
            print("Run %d" % run)
            answers = []  # List of counters
            for x in tqdm(chunks(inputs, batch_size)):
                answer_set = predict_self_consistency(task_description, x)
                result_counter = [Counter() for i in range(batch_size)]
                for chunk_no, ans_chunk in enumerate(chunks(answer_set, 5)):
                    preds = []
                    for x in ans_chunk:
                        if re.search("""The final answer is """, x):
                            preds.append(x[re.search("""The final answer is """, x).span(0)[-1] :])
                        else:
                            preds.append(x.strip())
                    for enum, pred in enumerate(ans_chunk):
                        # Only add to the counter if there is a legitimate answer
                        if re.search("""The final answer is """, pred):
                            result_counter[chunk_no].update(
                                [pred[re.search("""The final answer is """, x).span(0)[-1] :]]
                            )
                answers.extend(result_counter)
            preds = [x.most_common(1)[0][0] for x in answers]
            perf_array.append(substring_match(labels, preds))
        print("FS-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))
    else:
        perf_array = []
        runs = 5
        for run in range(runs):
            print("Run %d" % run)
            answers = []
            for x in tqdm(chunks(inputs, 10)):
                x = [ex.replace(" Answer:", "") for ex in x]
                answers.extend(predict(x))
            preds = [get_autocot_answer(x, answer_prompt="The final answer is ") for x in answers]
            perf_array.append(substring_match(labels, preds))
            print(perf_array)
        print("Auto-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))


def few_shot_cot(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):
    global few_shot_cot_prompt

    if strategy == "fixed":
        few_shot_cot_prompt = few_shot_cot_prompt
    elif strategy == "random":
        few_shot_cot_prompt = random_tasks(N=6)
    elif strategy == "similar":
        few_shot_cot_prompt = similar_tasks(task_description, io_pairs, N=6)
    elif strategy == "similar_auto_decomp":
        few_shot_cot_prompt = similar_auto_breakdowns(task_description, io_pairs, N=6)
    elif strategy == "llm_similar":
        few_shot_cot_prompt = llm_similar_tasks(task_name, task_description, io_pairs, N=6)

    interpreter = TopDownVisitorBeta(model_name=model_name, temperature=temperature)

    def predict(description, chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return gpt3(prompts)

    def predict_complete(description, chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        outputs = gpt3(prompts)
        completed_outputs = [
            interpreter.complete_program(prefix, output) for prefix, output in zip(prompts, outputs)
        ]
        return completed_outputs

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            answers.extend(predict_complete(task_description, x))
            pdb.set_trace()
        preds = [get_answer(x.strip()) for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def nl_program(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):
    global few_shot_cot_prompt

    if strategy == "fixed":
        few_shot_cot_prompt = few_shot_cot_prompt
    elif strategy == "random":
        few_shot_cot_prompt = random_tasks(N=5)
    elif strategy == "similar":
        few_shot_cot_prompt = similar_tasks(task_description, io_pairs, N=5)
    elif strategy == "similar_auto_decomp":
        few_shot_cot_prompt = similar_auto_breakdowns(task_description, io_pairs, N=5)
    elif strategy == "llm_similar":
        few_shot_cot_prompt = llm_similar_tasks(task_name, task_description, io_pairs, N=5)

    interpreter = TopDownVisitorBeta(model_name=model_name)

    def predict(description, chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nEdited:", "") for ex in x]
            prompts, answer = predict(task_description, x)
            new_answer = interpreter.batch_visit(prompts, answer)
            answers.extend(new_answer)
        preds = [get_answer(x) for x in answers]
        perf_array.append(substring_match(labels, preds))
    print("FS-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def nl_program(
    temperature=0.3,
    model_name="text-davinci-002",
    strategy="fixed",
    self_consistency=False,
):
    global few_shot_cot_prompt

    if strategy == "fixed":
        few_shot_cot_prompt = few_shot_cot_prompt
    elif strategy == "random":
        few_shot_cot_prompt = random_tasks(N=6)
    elif strategy == "similar":
        few_shot_cot_prompt = similar_tasks(task_description, io_pairs, N=6)
    elif strategy == "similar_auto_decomp":
        few_shot_cot_prompt = similar_auto_breakdowns(task_description, io_pairs, N=6)
    elif strategy == "llm_similar":
        few_shot_cot_prompt = llm_similar_tasks(task_name, task_description, io_pairs, N=6)

    interpreter = TopDownVisitorBeta(model_name=model_name)

    def predict(description, chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    def predict_self_consistency(description, chunk, n=5):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=n
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    perf_array = []
    runs = 3
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            prompts, answer = predict(task_description, x)
            new_answer = interpreter.batch_visit(prompts, answer)
            answers.extend(new_answer)
        preds = [get_answer(x.strip()) for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
        positive_calls = [
            int(len(stack_trace_list) >= 1) for stack_trace_list in interpreter.execution_details
        ]
        positive_rate = sum(positive_calls) / len(interpreter.execution_details)
    print("FS-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))
    print("Rate of affordance call", positive_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="astronomy")
    parser.add_argument("--task_name", type=str, default="astronomy")
    parser.add_argument(
        "--model_name",
        type=str,
        choices=[
            "text-davinci-002",
            "text-davinci-003",
            "code-davinci-002",
            "code-cushman-001",
            "davinci-codex-002-msft",
        ],
        default="text-davinci-002",
    )
    parser.add_argument("--temperature", type=float, default="0.3")
    parser.add_argument(
        "--inference_strategy",
        type=str,
        choices=[
            "dummy",
            "few_shot",
            "auto_cot",
            "cot_rollout",
            "few_shot_cot",
            "nl_program",
        ],
        default="few_shot",
    )
    parser.add_argument("--num_train_examples", type=int, default=10)
    parser.add_argument("--num_dev_examples", type=int, default=len(inputs))
    parser.add_argument("--self_consistency", default=False, action="store_true")
    parser.add_argument(
        "--selection_strategy",
        type=str,
        choices=["fixed", "random", "similar", "similar_auto_decomp", "llm_similar"],
        default="fixed",
    )

    args = parser.parse_args()

    print("Dataset statistics")
    print(task_description)
    print("Training examples:", len(train_inputs))
    print("Dev examples:", len(inputs))

    inputs = inputs[: args.num_dev_examples]
    labels = labels[: args.num_dev_examples]

    if args.inference_strategy == "few_shot":
        few_shot_prompt = get_few_shot_prompt(train_inputs, train_labels, n=args.num_train_examples)
        print("Length of few-shot prompt", len(tokenizer(few_shot_prompt)["input_ids"]))
        few_shot(args.num_train_examples, args.temperature, args.model_name)
    elif args.inference_strategy == "auto_cot":
        auto_cot(
            args.temperature,
            args.model_name,
            predict=True,
            use_corrected=False,
            self_consistency=False,
        )
    elif args.inference_strategy == "few_shot_cot":
        few_shot_cot(args.temperature, args.model_name, strategy=args.selection_strategy)
    elif args.inference_strategy == "nl_program":
        nl_program(
            args.temperature,
            args.model_name,
            self_consistency=args.self_consistency,
            strategy=args.selection_strategy,
        )
