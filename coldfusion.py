import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import faiss
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import os

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_model(model_name):
    if model_name == 'mpnet':
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    elif model_name == 'gte':
        tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
        model = AutoModel.from_pretrained("thenlper/gte-large")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model, tokenizer

def get_in_scope_queries(dataset):
    if dataset == 'banking77':
        queries = {
        'activate my card': 'How can I activate my new card?',
        'age limit': 'Is there an age limit to open a bank account with your service?',
        'apple pay or google pay': 'Do you support Apple Pay or Google Pay for transactions?',
        'atm support': 'Which ATMs can I use for withdrawals with your bank card?',
        'automatic top up': 'How can I set up automatic top-up for my account?',
        'balance not updated after bank transfer': 'My balance hasn’t updated after a bank transfer. What should I do?',
        'balance not updated after cheque or cash deposit': 'I deposited a cheque/cash, but my balance hasn’t updated. Why?',
        'beneficiary not allowed': 'Why am I unable to add a particular beneficiary to my account?',
        'cancel transfer': 'Can I cancel a transfer I initiated recently?',
        'card about to expire': 'How can I check when my card is about to expire?',
        'card acceptance': 'Which merchants or services accept your bank cards?',
        'card arrival': 'When can I expect the arrival of my new bank card?',
        'card delivery estimate': 'What is the estimated delivery time for a new bank card?',
        'card linking': 'How do I link my card to other accounts or services?',
        'card not working': 'My card is not working. What troubleshooting steps should I follow?',
        'card payment fee charged': 'Why was I charged a fee for a card payment?',
        'card payment not recognised': 'I made a card payment, but it\'s not recognized in my account. What should I do?',
        'card payment wrong exchange rate': 'I noticed a wrong exchange rate for a recent card payment. How can I address this?',
        'card swallowed': 'My card got swallowed by an ATM. What should I do now?',
        'cash withdrawal charge': 'Is there a charge for cash withdrawals using your bank card?',
        'cash withdrawal not recognised': 'I withdrew cash, but the transaction is not recognized in my account. What\'s the issue?',
        'change pin': 'How can I change the PIN for my bank card?',
        'compromised card': 'I suspect my card details are compromised. What should I do to secure my account?',
        'contactless not working': 'My contactless payment is not working. How can I fix this issue?',
        'country support': 'Which countries does your bank service support?',
        'declined card payment': 'My card payment was declined. What could be the reasons for this?',
        'declined cash withdrawal': 'Why was my cash withdrawal declined at the ATM?',
        'declined transfer': 'A transfer I attempted was declined. What steps should I take?',
        'direct debit payment not recognised': 'I have a direct debit payment not recognized in my account. What\'s the reason?',
        'disposable card limits': 'What are the limits for transactions with disposable virtual cards?',
        'edit personal details': 'How can I edit or update my personal details on the account?',
        'exchange charge': 'Is there a charge for currency exchange using your bank card?',
        'exchange rate': 'How can I check the current exchange rates for currencies?',
        'exchange via app': 'Can I perform currency exchange directly through the mobile app?',
        'extra charge on statement': 'I noticed an extra charge on my statement. Can you explain this?',
        'failed transfer': 'A transfer I initiated has failed. What could be the reasons?',
        'fiat currency support': 'Which fiat currencies are supported by your bank?',
        'get disposable virtual card': 'How can I obtain a disposable virtual card?',
        'get physical card': 'What is the process for obtaining a physical bank card?',
        'getting spare card': 'Can I get a spare or backup bank card?',
        'getting virtual card': 'How can I get a virtual bank card?',
        'lost or stolen card': 'My card is lost or stolen. What immediate steps should I take?',
        'lost or stolen phone': 'If I lose my phone, how can I secure my bank account?',
        'order physical card': 'How can I order a new physical bank card?',
        'passcode forgotten': 'I forgot my passcode. How can I recover or reset it?',
        'pending card payment': 'Why is a card payment showing as pending in my account?',
        'pending cash withdrawal': 'I have a pending cash withdrawal. When will it be processed?',
        'pending top up': 'My top-up is pending. When will it reflect in my account balance?',
        'pending transfer': 'How long does it take for a transfer to move from pending to completed status?',
        'pin blocked': 'My PIN got blocked. How can I unblock it?',
        'receiving money': 'How can I receive money into my account?',
        'Refund not showing up': 'I initiated a refund, but it\'s not showing up in my account. What should I do?',
        'request refund': 'How can I request a refund for a transaction?',
        'reverted card payment?': 'A card payment was reverted. What could be the reason for this?',
        'supported cards and currencies': 'Which types of cards and currencies does your bank support?',
        'terminate account': 'What is the process for terminating or closing my bank account?',
        'top up by bank transfer charge': 'Is there a charge for topping up my account via bank transfer?',
        'top up by card charge': 'Are there any charges for topping up my account using a bank card?',
        'top up by cash or cheque': 'Can I top up my account using cash or a cheque?',
        'top up failed': 'My top-up attempt failed. What could be the reasons for this?',
        'top up limits': 'Are there any limits on the amount I can top up into my account?',
        'top up reverted': 'Why was my top-up amount reverted? What should I do?',
        'topping up by card': 'How can I top up my account using a bank card?',
        'transaction charged twice': 'I noticed a transaction charged twice. How can I rectify this?',
        'transfer fee charged': 'Why was I charged a fee for a transfer?',
        'transfer into account': 'How can I transfer funds into my account from another bank?',
        'transfer not received by recipient': 'The recipient didn’t receive the funds I transferred. What should I do?',
        'transfer timing': 'What are the processing times for fund transfers?',
        'unable to verify identity': 'I am unable to verify my identity. What should I do?',
        'verify my identity': 'How can I verify my identity with your bank?',
        'verify source of funds': 'Why do I need to verify the source of funds in my account?',
        'verify top up': 'Why do I need to verify my top-up transactions?',
        'virtual card not working': 'My virtual card is not working. What steps should I take?',
        'visa or mastercard': 'Do you issue Visa or Mastercard for your bank cards?',
        'why verify identity': 'Why is identity verification necessary for using your bank services?',
        'wrong amount of cash received': 'I received the wrong amount of cash after a withdrawal. How can this be corrected?',
        'wrong exchange rate for cash withdrawal': 'The exchange rate for my recent cash withdrawal seems incorrect. What should I do?'
    }
    elif dataset == 'clinc_banking':
        queries = {
            'transactions': "Can you provide a list of my recent transactions?",
            'report_fraud': "I suspect fraudulent activity on my account, how can I report it?",
            'routing': "What is the bank's routing number for wire transfers?",
            'interest_rate': "What is the current interest rate on savings accounts?",
            'bill_balance': "What is the outstanding balance on my credit card bill?",
            'order_checks': "How can I order a new set of checks for my checking account?",
            'pin_change': "I need to change my PIN, how can I do that?",
            'pay_bill': "How do I set up automatic bill payments from my account?",
            'spending_history': "Can you provide a summary of my spending history for the last month?",
            'account_blocked': "Why is my account blocked, and how can I resolve this issue?"
        }
    elif dataset == 'clinc_credit_cards':
        queries = {
            'expiration_date': "Can you remind me of my credit card's expiration date?",
            'apr': "What is the current APR on my credit card?",
            'new_card': "How can I apply for a new credit card?",
            'redeem_rewards': "What are the options to redeem my credit card rewards?",
            'credit_score': "Could you provide information about my current credit score?",
            'card_declined': "Why was my credit card declined during the recent transaction?",
            'damaged_card': "My credit card got damaged, what should I do to get a replacement?",
            'credit_limit_change': "Can I request a change in my credit card's limit?",
            'international_fees': "What are the international transaction fees on my credit card?",
            'credit_limit': "What is my current credit card limit?"
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    for key, value in queries.items():
        queries[key] = [value]
    return queries

def encode_queries(queries, model, tokenizer, model_name):
    in_scope_topic_features = []
    with torch.no_grad():
        for key, value in tqdm(queries.items(), total=len(queries)):
            if model_name == 'gte':
                batch_dict = tokenizer(queries[key], max_length=512, padding=True, truncation=True, return_tensors='pt')
                outputs = model(**batch_dict)
                cur_occ = average_pool(outputs.last_hidden_state,
                                       batch_dict['attention_mask']).contiguous().cpu().numpy()
            else:
                batch_dict = tokenizer(queries[key], padding=True, truncation=True, return_tensors='pt')
                outputs = model(**batch_dict)
                cur_occ = mean_pooling(outputs, batch_dict['attention_mask'])
            in_scope_topic_features.append(cur_occ)
        in_scope_topic_features = np.array(in_scope_topic_features)
    return in_scope_topic_features

def main(args):
    np.random.seed(0)
    model_name = args.model
    dataset = args.dataset

    train_normal = np.load(f'./features/{dataset}/{model_name}/train.npy')
    train_id_oos = np.load(f'./features/{dataset}/{model_name}/id-oos-valid.npy')

    test_normal = np.load(f'./features/{dataset}/{model_name}/test.npy')
    test_id_oos = np.load(f'./features/{dataset}/{model_name}/id-oos-test.npy')

    model, tokenizer = get_model(model_name)
    queries = get_in_scope_queries(dataset)
    in_scope_topic_features = encode_queries(queries, model, tokenizer, model_name)

    anom_prec = args.anom_prec
    number_of_in_scope_topics = len(in_scope_topic_features)
    topk_perc = args.topk_prec

    trainset_1 = train_normal
    num_of_anom = np.round(len(trainset_1) * anom_prec).astype(np.int32)
    indices = np.random.permutation(len(train_id_oos))[:num_of_anom]
    trainset_2 = train_id_oos[indices]
    trainset = np.concatenate((trainset_1, trainset_2), 0)
    shuffle = np.random.permutation(len(trainset))
    trainset = trainset[shuffle]

    testset = np.concatenate((test_normal, test_id_oos), 0)
    labels = np.zeros(len(testset))
    labels[len(test_normal):] = 1
    in_scope_topic_features = np.concatenate(in_scope_topic_features, 0)

    all_aucs_zero = []
    all_aucs_occ = []
    all_aucs_coldfusion = []

    for i in range(0, len(trainset)):
        cur_trainset = trainset[:i + 1]

        if i >= 1 and anom_prec > 0:
            index = faiss.IndexFlatL2(cur_trainset.shape[1])
            index.add(in_scope_topic_features)
            D, _ = index.search(cur_trainset, 1)
            distances = np.mean(D, axis=1)
            indices = np.argsort(distances)
            topk = np.round(len(cur_trainset) * topk_perc).astype(np.int32)
            cur_trainset = cur_trainset[indices[:topk]].astype(np.float32)

        index = faiss.IndexFlatL2(cur_trainset.shape[1])
        index.add(in_scope_topic_features)
        D, I = index.search(cur_trainset, 1)
        I = I[:, 0]

        each_class_assignments = []
        for q in range(number_of_in_scope_topics):
            each_class_assignments.append([in_scope_topic_features[q][None]])
        for q in range(len(I)):
            each_class_assignments[I[q]].append(cur_trainset[q][None])
        for q in range(number_of_in_scope_topics):
            each_class_assignments[q] = np.median(each_class_assignments[q], 0)
        adapted_class_embeddings = np.concatenate(each_class_assignments, 0)

        k = 1

        index = faiss.IndexFlatL2(cur_trainset.shape[1])
        index.add(in_scope_topic_features)
        D, _ = index.search(testset, k)
        distances_zero = np.mean(D, axis=1)
        auc_zero = roc_auc_score(labels, distances_zero)

        index = faiss.IndexFlatL2(cur_trainset.shape[1])
        index.add(cur_trainset)
        D, _ = index.search(testset, k)
        distances = np.mean(D, axis=1)
        auc_dn2 = roc_auc_score(labels, distances)

        index = faiss.IndexFlatL2(cur_trainset.shape[1])
        index.add(adapted_class_embeddings)
        D, _ = index.search(testset, k)
        distances_coldfusion = np.mean(D, axis=1)
        auc_coldfusion = roc_auc_score(labels, distances_coldfusion)

        all_aucs_zero.append(auc_zero * 100)
        all_aucs_occ.append(auc_dn2 * 100)
        all_aucs_coldfusion.append(auc_coldfusion * 100)

    output_path = f'./figures/{dataset}/{model_name}/{anom_prec}/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    final_output_path = os.path.join(output_path, 'output.png')

    sns.set_style("whitegrid")
    num_of_steps = np.arange(len(trainset))
    plt.figure()
    plt.plot(num_of_steps, all_aucs_zero, label=f'ZS', color='tab:red')
    plt.plot(num_of_steps, all_aucs_occ, label=f'DN2', color='tab:purple')
    plt.plot(num_of_steps, all_aucs_coldfusion, label=f'ColdFusion', color='tab:blue')

    plt.xlabel('Number of queries', fontsize='xx-large')
    plt.ylabel('AUROC (\%)', fontsize='xx-large')
    leg = plt.legend(fontsize='xx-large')
    plt.savefig(final_output_path, dpi=600)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='banking77', choices=['banking77', 'clinc_banking', 'clinc_credit_cards'])
    parser.add_argument('--model', default='gte', choices=['gte', 'mpnet'])
    parser.add_argument('--anom_prec', type=float, default=0.05)
    parser.add_argument('--topk_prec', type=float, default=0.9)
    args = parser.parse_args()
    main(args)