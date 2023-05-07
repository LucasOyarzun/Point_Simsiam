# from .runner import run_net
from .runner_pretrain import run_net as pretrain_run_net
from .runner_finetune import run_net as finetune_run_net
from .runner_finetune import test_net as test_run_net
from .runner_test_svm import run_net_svm_modelnet40 as test_svm_run_net_modelnet40
from .runner_test_svm import run_net_svm_scan as test_svm_run_net_scan
from .runner_test_knn import run_net_knn_modelnet40 as test_knn_run_net_modelnet40
from .runner_test_knn import run_net_knn_scan as test_knn_run_net_scan
from .visualization import run_visualization
from .test_invariation import run_test_invariation