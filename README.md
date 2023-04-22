# Kmeans

Ce programme est un TP réalisé dans le cours "IA et apprentissage" de la licence 3 Informatique de Paris 8.

Ce dernier est une implémentation d'un Kmeans from scratch en python. Dans l'état actuel le dataset fournis par la classe "Dataset" produit plusieurs groupes de points.

Ces groupes sont ensuite concaténés pour être donner à notre classe Kmeans. Cette dernière est programmée de manière à pouvoir gérer un dataset à N dimensions.

Le Kmeans utilise une distance de Minkowski pour calculer les distances aux centroïdes.

Le Kmeans est paramétrable à travers deux arguments : le nombre de clusters recherchés et l'ordre de la distance de Minkowski.

Le programme produit une image du partitionnement à chaque tour du Kmeans. Finalement, il compile ces images pour produire une vidéo.


