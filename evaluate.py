import torch

from data.anchor_genration import generate_anchor, tag_anchor

def val(net, criterion, batch_num, using_cuda, logger, val_data):
    total_loss = 0
    total_cls_loss = 0
    total_v_reg_loss = 0
    total_o_reg_loss = 0

    for ind, (img, tag, filename) in enumerate(val_data):
        if ind > batch_num:
            break
        tensor_img = img.permute((0, 3, 1, 2))
        img = torch.squeeze(img, 0)
        if using_cuda:
            tensor_img = tensor_img.to(dtype=torch.float).cuda()
        else:
            tensor_img = tensor_img.to(dtype=torch.float)
        
        vertical_pred, score, side_refinement = net(tensor_img)
        del tensor_img
        positive = []
        negative = []
        vertical_reg = []
        side_refinement_reg = []

        try:
            for box in tag:
                gt_anchor = generate_anchor(img, box)
                positive1, negative1, vertical_reg1, side_refinement_reg1 = tag_anchor(gt_anchor, score, box)
                positive += positive1
                negative += negative1
                vertical_reg += vertical_reg1
                side_refinement_reg += side_refinement_reg1
        
        except:
            print("warning: img {} raise error".format(filename))
            batch_num -= 1
            continue

        if len(vertical_reg) == 0 or len(positive) == 0 or len(side_refinement_reg) == 0:
            batch_num -= 1
            continue

        loss, cls_loss, v_reg_loss, o_reg_loss = criterion(score, vertical_pred, side_refinement, positive, negative, vertical_reg, side_refinement_reg)

        total_loss += float(loss)
        total_cls_loss += float(cls_loss)
        total_v_reg_loss += float(v_reg_loss)
        total_o_reg_loss += float(o_reg_loss)
    
    logger.info('-------- Start evaluate ---------')
    logger.info('Evaluate loss: {0}'.format(total_loss / float(batch_num)))
    logger.info('Evaluate vertical regression loss: {0}'.format(total_v_reg_loss / float(batch_num)))
    logger.info('Evaluate side-refinement regression loss: {0}'.format(total_o_reg_loss / float(batch_num)))
    logger.info('-------- End evaluate ---------')

    return total_loss
