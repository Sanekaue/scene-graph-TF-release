import tensorflow as tf
import os
import shutil
from pprint import pprint

def rename(checkpoint_fname_src, checkpoint_fname_dst, replace_dict, add_prefix,
 dry_run):
    """
    To use this new and improved version of rename, don't run this from the command line.
    Start iPython (my personal fave REPL) and do like this:
  	rename_dict = {'edge_rnn/GRUCell/Candidate/Linear/Bias': 'edge_rnn/gru_cell/candidate/bias',
 				   'edge_rnn/GRUCell/Candidate/Linear/Matrix': 'edge_rnn/gru_cell/candidate/kernel',
				   'edge_rnn/GRUCell/Gates/Linear/Bias': 'edge_rnn/gru_cell/gates/bias',
				   'edge_rnn/GRUCell/Gates/Linear/Matrix': 'edge_rnn/gru_cell/gates/kernel',
				   'vert_rnn/GRUCell/Candidate/Linear/Bias': 'vert_rnn/gru_cell/candidate/bias',
 				   'vert_rnn/GRUCell/Candidate/Linear/Matrix': 'vert_rnn/gru_cell/candidate/kernel',
				   'vert_rnn/GRUCell/Gates/Linear/Bias': 'vert_rnn/gru_cell/gates/bias',
				   'vert_rnn/GRUCell/Gates/Linear/Matrix': 'vert_rnn/gru_cell/gates/kernel'
				  }
	 ckpt_src = "/path/to/existing/checkpoint/file"
	 ckpt_dst = "/path/to/output/directory/"
	 rename(ckpt_src, ckpt_dst, rename_dict, "", False)

    """
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_fname_src):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_fname_src, var_name)

            # Set the new name
            new_name = var_name
            if var_name in replace_dict:
                new_name = new_name.replace(var_name, replace_dict[var_name])
                if add_prefix:
                    new_name = add_prefix + new_name
                if dry_run:
                    print('%s would be renamed to %s.' % (var_name,
                                                          new_name))
                else:
                    print('Renaming %s to %s.' % (var_name, new_name))
            # Create the variable, potentially renaming it
            var = tf.Variable(var, name=new_name)

        if not dry_run:
            # Save the variables
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.save(sess, checkpoint_fname_dst)



shutil.move('checkpoints',  'old_checkpoints')
fname = 'dual_graph_vrd_final_iter2.ckpt'
fname_src = os.path.join(os.path.abspath('.'), 'old_checkpoints', fname)
fname_dst = os.path.join(os.path.abspath('.'), 'checkpoints', fname)
rename_dict = {'edge_rnn/GRUCell/Candidate/Linear/Bias': 'edge_rnn/gru_cell/candidate/bias',
     				   'edge_rnn/GRUCell/Candidate/Linear/Matrix': 'edge_rnn/gru_cell/candidate/kernel',
    				   'edge_rnn/GRUCell/Gates/Linear/Bias': 'edge_rnn/gru_cell/gates/bias',
    				   'edge_rnn/GRUCell/Gates/Linear/Matrix': 'edge_rnn/gru_cell/gates/kernel',
    				   'vert_rnn/GRUCell/Candidate/Linear/Bias': 'vert_rnn/gru_cell/candidate/bias',
     				   'vert_rnn/GRUCell/Candidate/Linear/Matrix': 'vert_rnn/gru_cell/candidate/kernel',
    				   'vert_rnn/GRUCell/Gates/Linear/Bias': 'vert_rnn/gru_cell/gates/bias',
    				   'vert_rnn/GRUCell/Gates/Linear/Matrix': 'vert_rnn/gru_cell/gates/kernel'
    				  }

rename(fname_src, fname_dst, rename_dict, "", False)
shutil.copy(fname_dst+'.index', fname_dst)


with tf.Session() as sess:
  vars = tf.contrib.framework.list_variables(fname_dst)

pprint(vars)
assert vars == [('bbox_pred/biases', [604]),
 ('bbox_pred/weights', [512, 604]),
 ('cls_score/biases', [151]),
 ('cls_score/weights', [512, 151]),
 ('conv1_1/biases', [64]),
 ('conv1_1/weights', [3, 3, 3, 64]),
 ('conv1_2/biases', [64]),
 ('conv1_2/weights', [3, 3, 64, 64]),
 ('conv2_1/biases', [128]),
 ('conv2_1/weights', [3, 3, 64, 128]),
 ('conv2_2/biases', [128]),
 ('conv2_2/weights', [3, 3, 128, 128]),
 ('conv3_1/biases', [256]),
 ('conv3_1/weights', [3, 3, 128, 256]),
 ('conv3_2/biases', [256]),
 ('conv3_2/weights', [3, 3, 256, 256]),
 ('conv3_3/biases', [256]),
 ('conv3_3/weights', [3, 3, 256, 256]),
 ('conv4_1/biases', [512]),
 ('conv4_1/weights', [3, 3, 256, 512]),
 ('conv4_2/biases', [512]),
 ('conv4_2/weights', [3, 3, 512, 512]),
 ('conv4_3/biases', [512]),
 ('conv4_3/weights', [3, 3, 512, 512]),
 ('conv5_1/biases', [512]),
 ('conv5_1/weights', [3, 3, 512, 512]),
 ('conv5_2/biases', [512]),
 ('conv5_2/weights', [3, 3, 512, 512]),
 ('conv5_3/biases', [512]),
 ('conv5_3/weights', [3, 3, 512, 512]),
 ('edge_rnn/gru_cell/candidate/bias', [512]),
 ('edge_rnn/gru_cell/candidate/kernel', [1024, 512]),
 ('edge_rnn/gru_cell/gates/bias', [1024]),
 ('edge_rnn/gru_cell/gates/kernel', [1024, 1024]),
 ('edge_unary/biases', [512]),
 ('edge_unary/weights', [4096, 512]),
 ('fc6/biases', [4096]),
 ('fc6/weights', [25088, 4096]),
 ('fc7/biases', [4096]),
 ('fc7/weights', [4096, 4096]),
 ('in_edge_w_fc/biases', [1]),
 ('in_edge_w_fc/weights', [1024, 1]),
 ('obj_vert_w_fc/biases', [1]),
 ('obj_vert_w_fc/weights', [1024, 1]),
 ('out_edge_w_fc/biases', [1]),
 ('out_edge_w_fc/weights', [1024, 1]),
 ('rel_fc6/biases', [4096]),
 ('rel_fc6/weights', [25088, 4096]),
 ('rel_fc7/biases', [4096]),
 ('rel_fc7/weights', [4096, 4096]),
 ('rel_score/biases', [51]),
 ('rel_score/weights', [512, 51]),
 ('sub_vert_w_fc/biases', [1]),
 ('sub_vert_w_fc/weights', [1024, 1]),
 ('vert_rnn/gru_cell/candidate/bias', [512]),
 ('vert_rnn/gru_cell/candidate/kernel', [1024, 512]),
 ('vert_rnn/gru_cell/gates/bias', [1024]),
 ('vert_rnn/gru_cell/gates/kernel', [1024, 1024]),
 ('vert_unary/biases', [512]),
 ('vert_unary/weights', [4096, 512])]
