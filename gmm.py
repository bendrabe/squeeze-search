import numpy as np

SIM_LEN = 1000
DIM_EVAL = 50000
DIM_TEST = 10000

mean_eval_l = []
mean_test_l = []
var_eval_l = []
var_test_l = []

with open("trials_random/results.txt", "r") as f:
    for line in f:
        if "TRIAL" in line:
            continue

        trial = line.split(',')
        mean_eval_i = float(trial[8])
        mean_test_i = float(trial[9])

        mean_eval_l.append(mean_eval_i)
        mean_test_l.append(mean_test_i)
        var_eval_l.append( (mean_eval_i*(1-mean_eval_i))/(DIM_EVAL-1) )
        var_test_l.append( (mean_test_i*(1-mean_test_i))/(DIM_TEST-1) )

mean_eval = np.array(mean_eval_l)
mean_test = np.array(mean_test_l)
var_eval = np.array(var_eval_l)
var_test = np.array(var_test_l)

x = []
y = []

# number of trials per experiment
for s in [1,2,4,8,16,32,64,128]:
    # number of experiments
    N = 128//s
    # for each experiment
    for n in range(N):
        w = np.zeros(s)
        i_start = n*s
        # perform SIM_LEN simulations
        for _ in range(SIM_LEN):
            z = []
            for j in range(s):
                z.append( np.random.normal(mean_eval[i_start+j], np.sqrt(var_eval[i_start+j])) )
            z_np = np.array(z)
            w[ np.argmax(z_np) ] += 1
        w = w/SIM_LEN
        print("w: " + str(w))
        print("w_sum: " + str(np.sum(w)))
        mean_n = 0.0
        var_n = 0.0
        for j in range(s):
            mean_n += w[j]*mean_test[i_start+j]
        for j in range(s):
            var_n += w[j]*(mean_test[i_start+j] ** 2 + var_test[i_start+j] ** 2)
        var_n -= (mean_n ** 2)
        x.append(str(s))
        y.append(mean_n)

print("x = " + str(x))
print("y = " + str(y))
print("conf_max = " + str(mean_n + 1.96*np.sqrt(var_n)))
print("conf_min = " + str(mean_n - 1.96*np.sqrt(var_n)))
