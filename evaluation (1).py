from gru4rec_pytorch import SessionDataIterator
import torch

@torch.no_grad()
def batch_eval(gru, test_data, cutoff=[20], batch_size=512, mode='conservative', item_key='ItemId', session_key='SessionId', time_key='Time', return_predictions=False):
    if gru.error_during_train: 
        raise Exception('Attempting to evaluate a model that wasn\'t trained properly (error_during_train=True)')
    recall = dict()
    mrr = dict()
    for c in cutoff:
        recall[c] = 0
        mrr[c] = 0
    H = []
    for i in range(len(gru.layers)):
        H.append(torch.zeros((batch_size, gru.layers[i]), requires_grad=False, device=gru.device, dtype=torch.float32))
    n = 0
    predictions = []
    reset_hook = lambda n_valid, finished_mask, valid_mask: gru._adjust_hidden(n_valid, finished_mask, valid_mask, H)
    data_iterator = SessionDataIterator(test_data, batch_size, 0, 0, 0, item_key, session_key, time_key, device=gru.device, itemidmap=gru.data_iterator.itemidmap)
    for in_idxs, out_idxs in data_iterator(enable_neg_samples=False, reset_hook=reset_hook):
        for h in H: h.detach_()
        O = gru.model.forward(in_idxs, H, None, training=False)
        oscores = O.T
        tscores = torch.diag(oscores[out_idxs])
        if mode == 'standard': ranks = (oscores > tscores).sum(dim=0) + 1
        elif mode == 'conservative': ranks = (oscores >= tscores).sum(dim=0)
        elif mode == 'median':  ranks = (oscores > tscores).sum(dim=0) + 0.5*((oscores == tscores).dim(axis=0) - 1) + 1
        else: raise NotImplementedError
        for c in cutoff:
            recall[c] += (ranks <= c).sum().cpu().numpy()
            mrr[c] += ((ranks <= c) / ranks.float()).sum().cpu().numpy()
        if return_predictions:
            for i in range(len(out_idxs)):
                pred_items = oscores[:, i].topk(k=max(cutoff)).indices.cpu().numpy()  # Obtener predicciones
                actual_item = out_idxs[i].item()  # Ítem real de la sesión
                predictions.append((data_iterator.session_ids[i], pred_items, actual_item))
        
        n += O.shape[0]
    for c in cutoff:
        recall[c] /= n
        mrr[c] /= n

    if return_predictions:
        return recall,mrr, predictions
    print("se paso jajajaj")
        
    return recall, mrr
