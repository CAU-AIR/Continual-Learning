def cycle(loader):
    while True:
        for batch in loader:
            yield batch