# Étude des modèles : équations, rôle des hyperparamètres, hypothèses testables

Notation : fenêtre $x\in\mathbb{R}^{L}$ (L = 360 par défaut ; 216 ou 280 en ablation), étiquette $y\in\{0,1\}$ (1 = anormal), logit $z=f_\theta(x)$, probabilité $\hat p=\sigma(z)=1/(1+e^{-z})$.

## 1. MLP
$h^{(0)}=[x;\,r]$ (r = 5 caractéristiques RR standardisées, optionnel), $h^{(l)}=\phi(W^{(l)}h^{(l-1)}+b^{(l)})$, $z=w^\top h^{(K)}+b$.
- Aucun biais inductif de translation : un décalage de $s$ échantillons du QRS change totalement l'entrée. Conséquence attendue : plus sensible à la précision du centrage R, moins robuste au décalage temporel que le CNN.
- Hyperparamètres étudiés : profondeur/largeur (`mlp_a/b/c`), activation (ReLU, LeakyReLU, GELU), dropout, BatchNorm.
- Hypothèse H-MLP : à fenêtre centrée sur R, le MLP est compétitif avec le CNN en PR-AUC, mais perd plus quand le décalage est augmenté (test : augmentation `temporal_shift`).

## 2. CNN 1D
Convolution (corrélation croisée) : $h_{c'}[t]=\sum_{c=1}^{C}\sum_{k=0}^{K-1}W_{c',c}[k]\,h_c[t+k-\lfloor K/2\rfloor]+b_{c'}$, suivie de BN, activation, max-pool (ou stride). Partage de poids : $C'CK$ paramètres par couche, indépendants de $L$. Champ réceptif de la couche $l$ : $R_l=R_{l-1}+(K_l-1)\prod_{j<l}s_j$ (s = stride/pooling cumulé). Tête : global average pooling $g_c=\frac1T\sum_t h_c[t]$ (invariante à la translation, utilisée pour la CAM) ou aplatissement + dense (sensible à la position).
- Variantes : noyau 3/5/7, multi-échelle (3,5,7 concaténés), 2 vs 3 blocs, `pool=stride` vs max-pool, BN on/off, dropout, tête GAP vs flatten, largeur (`cnn_s`, `cnn_m`, `fcn`).
- Hypothèse H-CNN : le CNN apprend des motifs morphologiques locaux (largeur QRS, onde T, extrasystole ventriculaire) et dépasse le MLP en PR-AUC sur le jeu de validation.

## 3. CNN-LSTM
Le CNN produit une séquence $a_1,\dots,a_T$ ($T=L/2^{B}$ ; $T=90$ pour 2 blocs, L = 360), le LSTM la traite :
$$
\begin{aligned}
i_t&=\sigma(W_i a_t+U_i h_{t-1}+b_i),\quad f_t=\sigma(W_f a_t+U_f h_{t-1}+b_f),\quad o_t=\sigma(W_o a_t+U_o h_{t-1}+b_o)\\
\tilde c_t&=\tanh(W_c a_t+U_c h_{t-1}+b_c),\quad c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t,\quad h_t=o_t\odot\tanh(c_t)
\end{aligned}
$$
Le chemin $c_{t-1}\to c_t$ est additif, ce qui limite l'atténuation du gradient : $\partial c_t/\partial c_{t-1}=\operatorname{diag}(f_t)+\dots$. Le gradient sur la mémoire reste stable tant que $f_t\approx1$ ; il n'est pas garanti (explosion possible) → `clip_grad = 1.0` par défaut pour cette famille.
- Agrégation de la séquence : dernier état $h_T$, moyenne $\frac1T\sum_t h_t$, ou attention $\alpha=\operatorname{softmax}_t(v^\top h_t)$, $\sum_t\alpha_t h_t$ ; LSTM bidirectionnel (concaténation $\overrightarrow{h}_T,\overleftarrow{h}_1$) ; 1 ou 2 couches ; `rnn_dropout` agit entre couches empilées seulement (inopérant à 1 couche : cas enregistré, pas d'effet attendu).
- Hypothèse H-CNNLSTM (formulation de NS) : le CNN apprend des structures morphologiques locales, le LSTM modélise l'évolution temporelle de la séquence de caractéristiques ; **on ne présume pas** que cela améliore la généralisation par rapport au CNN seul. Test : comparaison appariée CNN vs CNN-LSTM aux mêmes seeds (42, 43, 44), même prétraitement, même perte, même budget d'époques ; mesure : PR-AUC / F1 validation (sélection) puis test (final), plus coût (paramètres, temps d'époque, latence). Un écart inférieur à la dispersion entre seeds n'est pas une amélioration. Risque noté : avec un battement isolé de 216 points et T = 54, la dépendance longue est faible ; un gain, s'il existe, peut venir de la capacité ajoutée plutôt que de la modélisation séquentielle (à contrôler en comparant à nombre de paramètres comparable).

## 4. Pertes
Avec $p=\hat p$, $p_t=yp+(1-y)(1-p)$ :
- BCE : $\mathcal L=-[y\log p+(1-y)\log(1-p)]$ ; gradient sur le logit : $\partial\mathcal L/\partial z=p-y$ (borné, ne s'annule pas quand le modèle est confiant et faux).
- BCE pondérée : poids $w_+=n_-/n_+$ sur les positifs (calculé sur train) ; déplace l'optimum de Bayes vers un seuil plus bas ; équivaut à rééchantillonner les classes dans la perte.
- Focal (Lin et al. 2017) : $-\alpha_t(1-p_t)^\gamma\log p_t$ ; $\gamma=0$ redonne la BCE ; $\gamma$ réduit le poids des exemples faciles ; la probabilité sortie est moins bien calibrée.
- Lissage d'étiquettes : $y'=y(1-\varepsilon)+\varepsilon/2$ ; limite la sur-confiance.
- MSE sur $\sigma(z)$ (**ablation, inadaptée**) : $\partial\mathcal L/\partial z=2(p-y)\,p(1-p)$ ; le facteur $p(1-p)\to0$ quand la sortie sature, donc un exemple confiant et faux apprend très lentement. Résultat attendu : pire convergence que la BCE ; à vérifier sur les courbes d'apprentissage, non affirmé.

## 5. Optimisation et régularisation
- SGD + Nesterov : $v\leftarrow\mu v+g$, $\theta\leftarrow\theta-\eta(g+\mu v)$.
- Adam (Kingma & Ba) : $m,v$ moments d'ordre 1/2 corrigés ; avec `weight_decay`, PyTorch ajoute $\lambda\theta$ au gradient (**L2 couplé**), donc la pénalité est divisée par $\sqrt{\hat v}$ : la régularisation effective dépend du gradient. AdamW (Loshchilov & Hutter) applique $\theta\leftarrow\theta-\eta\lambda\theta$ **découplé**. Les deux ne sont pas équivalents avec Adam (équivalents avec SGD, à un changement de $\lambda$ près).
- Schedulers : `plateau` (×0,5 après 2 époques sans gain de PR-AUC val), `cosine` ($\eta_t=\frac{\eta_0}{2}(1+\cos\frac{\pi t}{T})$).
- Gradient clipping : $g\leftarrow g\min(1,c/\|g\|)$.
- Early stopping : meilleur PR-AUC validation, patience 5 ; le nombre d'époques réel est enregistré.

## 6. Déséquilibre (prévalence ≈ 26 %)
Options comparées : perte pondérée, focal, échantillonnage pondéré (`sampler=weighted`, combiné avec BCE non pondérée pour ne pas compenser deux fois), augmentation. Avec ≈ 26 % d'anormaux le déséquilibre est modéré ; l'effet attendu est surtout un déplacement du compromis sensibilité/spécificité, récupérable aussi par le seuil → c'est pourquoi PR-AUC (indépendante du seuil) est le critère de sélection.

## 7. Seuils et erreurs
Trois seuils déterminés sur validation : 0,5 ; max-F1 ; plus haut seuil avec sensibilité ≥ 0,90. Contexte clinique : un faux négatif (arythmie manquée) est plus coûteux qu'un faux positif ; le rapport donne les deux taux par enregistrement et par symbole (`test_errors_by_record.csv`, `test_errors_by_symbol.csv`).

## 8. Interprétabilité
LR : coefficients standardisés. RF : importance par permutation (PR-AUC validation). Profonds : gradients intégrés (ligne de base nulle, 20 pas) sur TP/TN/FP/FN ; CAM pour CNN à tête GAP (`CNN1D.cam`). Les attributions sont descriptives, pas causales.
