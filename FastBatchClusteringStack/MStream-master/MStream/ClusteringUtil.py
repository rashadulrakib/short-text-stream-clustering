def findTargetClusters(document, f_zs):
    targetClusters = []

    for f in document.text:
        if f not in f_zs:
            continue
        targetClusters.extend(f_zs[f])

    return set(targetClusters)
