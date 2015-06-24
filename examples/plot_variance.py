__author__ = 'kcx'

from examples.variance import POOL_SIZE, REPETITIONS, DIMENSION
import matplotlib.pyplot as plt
from numpy import sqrt
from numpy import load

ft = load('./results/variance.npy')
ft = ft.astype('float')

pvalues = ft[:, :, 0]

plt.subplot(1, 2, 1)


def confidence_intervals(phat, sgn, n=POOL_SIZE*REPETITIONS, z=1.96):
    return phat + sgn * z * sqrt(phat * (1 - phat) / float(n))



plt.xticks(rotation=70)

print pvalues
for p_values_for_dim, label, c in zip(pvalues.transpose(),  ['Analytic Mean Embeddings CF', 'Smooth CF'],['b-', 'g-']):
    plt.plot(DIMENSION, p_values_for_dim, c, label=label, linewidth=2)
    plt.fill_between(DIMENSION, confidence_intervals(p_values_for_dim, -1),
                     confidence_intervals(p_values_for_dim, 1), alpha=0.2, color=c[0])

plt.tick_params(axis='both', which='major', labelsize=16)
plt.xlabel('dimensions', fontsize=24)
plt.ylabel('test power', fontsize=24)

plt.title('Variance', fontsize=24)
plt.ylim([-0.04, 0.6])


plt.legend(prop={'size':25},loc='best')

plt.show()