import re
import time

import datasets
import numpy as np
from tqdm import tqdm
from utils import OpenAIModel, cache_dir, chunks, get_subset, propose_decomposition

d = datasets.load_dataset("bigbench", "movie_dialog_same_or_different", cache_dir=cache_dir)
inputs = d["validation"]["inputs"][:500]
# inputs = [x.split('\n')[0] for x in inputs]
labels = d["validation"]["targets"][:500]
labels = [l[0] for l in labels]


def exact_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() == predict.lower():
            correct += 1
        count += 1
    return (1.0 * correct) / count


def movie_dialogue():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002", max_length=200, quote="---", n=1)
        prompts = [
            """The following is a conversation between two people, but the transcript doesn't mark who said what: Look, it's Dr. Tom. Hey, Dr. Tom! Who's Dr. Tom? My chiropractor! ----- In the preceding conversation, were the sentences "Look, it's Dr. Tom." and "Hey, Dr. Tom!" said by the same or different individuals?
Answer:
same
----
The following is a conversation between two people, but the transcript doesn't mark who said what: Told you I'm fine! How many do you see? What?! Fuck off. Save yourself. You don't feel cold? It's a spring day... ----- In the preceding conversation, were the sentences "Save yourself." and "You don't feel cold?" said by the same or different individuals?
Answer:
different
----
The following is a conversation between two people, but the transcript doesn't mark who said what: We've broken out, oh, the blessed freedom of it all! Eh, have you got a nail file, these handcuffs are killing me. I was framed. I was innocent. Will you stop it! Sorry to disturb you, miss... ----- In the preceding conversation, were the sentences "I was innocent." and "Will you stop it!" said by the same or different individuals?
Answer:
different
----
The following is a conversation between two people, but the transcript doesn't mark who said what: You made it up. Uh-huh. You said you wanted fireworks. ----- In the preceding conversation, were the sentences "Uh-huh." and "You said you wanted fireworks." said by the same or different individuals?
Answer:
same
----
The following is a conversation between two people, but the transcript doesn't mark who said what: There you are. Who was that boy? An old friend. ----- In the preceding conversation, were the sentences "Who was that boy?" and "An old friend." said by the same or different individuals?
Answer:
different
----
The following is a conversation between two people, but the transcript doesn't mark who said what: Come on -- I'm going to storm into his office in front of everybody in the afternoon and then that night I'm going to kill him? I'd have to be really dumb to do that. Going after him before gets you off the hook for killing him that's your alibi. ----- In the preceding conversation, were the sentences "I'd have to be really dumb to do that." and "Going after him before gets you off the hook for killing him that's your alibi." said by the same or different individuals?
Answer:
different
----
The following is a conversation between two people, but the transcript doesn't mark who said what: So you do fantasize? Yes. About who? I fantasized about you. About me? Yes. ----- In the preceding conversation, were the sentences "So you do fantasize?" and "Yes." said by the same or different individuals?
Answer:
different
----
The following is a conversation between two people, but the transcript doesn't mark who said what: Oh, yeah. Damn. My watch is busted. Hey, Rookie. Be cool. Just stay with me. This is what we do. I seem nervous, huh? ----- In the preceding conversation, were the sentences "Oh, yeah." and "Damn." said by the same or different individuals?
Answer:
same
----
The following is a conversation between two people, but the transcript doesn't mark who said what: I'm afraid they've become a bit... over-cautious. Our American friends. What happens to the schedule? We must follow it. But will they? ----- In the preceding conversation, were the sentences "What happens to the schedule?" and "We must follow it." said by the same or different individuals?
Answer:
different
----
The following is a conversation between two people, but the transcript doesn't mark who said what: I encouraged you to come here. My fault as much as yours. I was...crazy...desperate. I took it out on you. I didn't mean it. I know what she sees in you. You're kind and you're brave. If I ever get out of you, I'll be glad to call you my friend. ----- In the preceding conversation, were the sentences "You're kind and you're brave." and "If I ever get out of you, I'll be glad to call you my friend." said by the same or different individuals?
Answer:
same
----
%s"""
            % x
            for x in chunk
        ]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            answers.extend(predict(x))
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(labels, preds))
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


# movie_dialogue()


def human_decomposition():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002", max_length=1000, quote="---", n=1)
        prompts = [
            """The following are conversations between two or more people, but the transcript doesn't mark who said what. Your task is to figure out if two dialogues were spoken by the same or different person. To do this task, break down the conversation into dialogues and annotate each dialogue with a speaker. To do so, you can use speaker names like Alice, Bob, Charlie, Dean, etc.
Look, it's Dr. Tom. Hey, Dr. Tom! Who's Dr. Tom? My chiropractor! ----- In the preceding conversation, were the sentences "Look, it's Dr. Tom." and "Hey, Dr. Tom!" said by the same or different individuals?
Dialogues:
Look, it's Dr. Tom.
Hey, Dr. Tom!
Who's Dr. Tom?
Speakers
Alice: Look, it's Dr. Tom.
Alice: Hey, Dr. Tom!
Bob: Who's Dr. Tom?

Who spoke the selected sentences? Alice, Alice
Were the sentences spoken by the same or different persons? same
----
Told you I'm fine! How many do you see? What?! Fuck off. Save yourself. You don't feel cold? It's a spring day... ----- In the preceding conversation, were the sentences "Save yourself." and "You don't feel cold?" said by the same or different individuals?
Dialogues:
Told you I'm fine!
How many do you see?
What?!
Fuck off.
Save yourself
You don't feel cold?
It's a spring day...
Speakers
Alice: Told you I'm fine!
Bob: How many do you see?
Alice: What?!
Alice: Fuck off.
Alice: Save yourself
Bob: You don't feel cold?
Alice:  It's a spring day...

Who spoke the selected sentences? Alice, Bob
Were the sentences spoken by the same or different persons? different
----
We've broken out, oh, the blessed freedom of it all! Eh, have you got a nail file, these handcuffs are killing me. I was framed. I was innocent. Will you stop it! Sorry to disturb you, miss... ----- In the preceding conversation, were the sentences "I was innocent." and "Will you stop it!" said by the same or different individuals?
Dialogues:
We've broken out, oh, the blessed freedom of it all!.
Eh, have you got a nail file, these handcuffs are killing me.
I was framed.
I was innocent
Will you stop it!
Sorry to disturb you, miss...
Speakers:
Bob: We've broken out, oh, the blessed freedom of it all!
Alice: Eh, have you got a nail file, these handcuffs are killing me.
Bob: I was framed.
Bob: I was innocent.
Alice: Will you stop it!
Bob: Sorry to disturb you, miss...

Who spoke the selected sentences? Bob, Alice
Were the sentences spoken by the same or different persons? different
----
You made it up. Uh-huh. You said you wanted fireworks. ----- In the preceding conversation, were the sentences "Uh-huh." and "You said you wanted fireworks." said by the same or different individuals?
Dialogues:
You made it up.
Uh-huh.
You said you wanted fireworks.
Speakers
Alice: You made it up.
Bob: Uh-huh.
Bob: You said you wanted fireworks.

Who spoke the selected sentences? Bob, Bob
Were the sentences spoken by the same or different persons? same
----
There you are. Who was that boy? An old friend. ----- In the preceding conversation, were the sentences "Who was that boy?" and "An old friend." said by the same or different individuals?
Dialogues:
There you are.
Who was that boy?
An old friend.
Speakers
Alice: There you are.
Bob: Who was that boy?
Alice: An old friend.

Who spoke the selected sentences? Bob, Alice
Were the sentences spoken by the same or different persons? different
----
I encouraged you to come here. My fault as much as yours. I was...crazy...desperate. I took it out on you. I didn't mean it. I know what she sees in you. You're kind and you're brave. If I ever get out of you, I'll be glad to call you my friend. ----- In the preceding conversation, were the sentences "You're kind and you're brave." and "If I ever get out of you, I'll be glad to call you my friend." said by the same or different individuals?
Dialogues:
I encouraged you to come here.
My fault as much as yours.
I was...crazy...desperate.
I took it out on you.
I didn't mean it.
I know what she sees in you.
You're kind and you're brave.
If I ever get out of you, I'll be glad to call you my friend.
Speakers
Alice: I encouraged you to come here.
Alice: My fault as much as yours.
Bob: I was...crazy...desperate.
Bob: I took it out on you.
Bob: I didn't mean it.
Alice: I know what she sees in you.
Alice: You're kind and you're brave.
Alice: If I ever get out of you, I'll be glad to call you my friend.

Who spoke the selected sentences? Alice, Alice
Were they spoken by the same or different persons? same
----
%s"""
            % x
            for x in chunk
        ]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            # Preprocess x to remove chunks
            x = [
                inp.replace(
                    "The following is a conversation between two people, but the transcript doesn't mark who said what: ",
                    "",
                ).replace("Answer:", "")
                for inp in x
            ]
            answers.extend(predict(x))
        preds = [x.strip().lower().split()[-1] for x in answers]
        perf_array.append(exact_match(labels, preds))
        time.sleep(30)
    print("Human decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


human_decomposition()


def automatic_decomposition():
    decomp_prompt = "The task is to figure out if two dialogues in a movie script snippet were spoken by the same or different person. I want to break this task into individual steps."
    io_pairs = """"""
    decompositions = propose_decomposition(decomp_prompt, io_pairs, 10)

    for decomp in decompositions:
        print(decomp)
        print("----")

    def get_list_reversal_fn(decomposition, batch_size=10):
        decomposition = "1." + decomposition
        last_n = int(re.findall(r"(\d+)\.", decomposition)[-1])

        #     decomposition += '\n%s. Output YES if there is an anachronism, and NO otherwise' % (last_n + 1)
        def decomposition_fn(sentences):
            gpt3 = OpenAIModel(model="text-davinci-002", max_length=1000, quote="---", n=1)
            out = []
            for chunk in chunks(sentences, batch_size):
                prompts = [
                    """The task is to figure out if two dialogues in a movie script snippet were spoken by the same or different person. Using the following steps will help.
Steps:
%s
----
%s
How did you arrived at this answer step-wise. Finally, output the words "same" or "different"."""
                    % (decomposition, x)
                    for x in chunk
                ]
                out.extend(gpt3(prompts))
            return out

        return decomposition_fn

    labs, subset = get_subset(inputs, labels=labels, n=len(inputs))
    preds = []
    pps = []
    accs = []
    for z, decomposition in enumerate(decompositions):
        print("Decomposition", z)
        fn = get_list_reversal_fn(decomposition, batch_size=20)
        this_preds = fn(subset)
        #     pp = np.array([1 if 'contains an anachronism' in x.lower() else 0 for x in this_preds])
        pp = [x.strip() for x in this_preds]
        # perf_array.append(exact_match(labels, preds))
        preds.append(this_preds)
        pps.append(pp)
        accs.append(exact_match(labs, pp))
        time.sleep(30)
    print(accs)
    print("Automatic decomposition Performance:")
    print("Mean", np.mean(accs))
    print("Std. Dev", np.std(accs))


# automatic_decomposition()
