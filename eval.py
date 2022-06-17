from datasets import load_metric

rouge = load_metric('rouge')
meteor = load_metric('meteor')
bleu = load_metric('bleu')
bertscore = load_metric('bertscore')


def rougeL_f1_scorer(candidate, reference):
    score = rouge.compute(predictions=[candidate], references=[reference], use_aggregator=False)
    return score['rougeL'][0].fmeasure


def meteor_scorer(candidate, reference):
    score = meteor.compute(predictions=[candidate], references=[reference])
    return score['meteor']


def bleu_scorer(candidate, reference):
    score = bleu.compute(predictions=[candidate.split()], references=[[reference.split()]])
    return score  # TODO: return number


def bert_scorer(candidate, reference, model_type='???'):
    score = bertscore.compute(predictions=[candidate], references=[reference], model_type=model_type)
    return score['f1'][0]


def score_paraphrases_average(paraphrases, ref_paraphrases, scorer):
    total_score = 0

    for para in paraphrases:
        best_match, best_score = score_single_paraphrase(para, ref_paraphrases, scorer)
        total_score += best_score

    average_score = total_score / len(paraphrases)

    return average_score


def score_single_paraphrase(paraphrase, ref_paraphrases, scorer):
    text = paraphrase.lstrip()  # strip leading spaces

    best_score = 0
    best_match = ""
    for ref_paraphrase in ref_paraphrases:
        score = scorer(text, ref_paraphrase)

        if score > best_score:
            best_score = score
            best_match = ref_paraphrase

    return best_match, best_score
