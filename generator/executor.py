from generator import Generator

if __name__ == '__main__':
    generator = Generator(
        n_items=10,
        bin_size=[10, 10, 10],
    )

    generator.generate(seed=0, shuffle=False, verbose=True)
    generator.generate(seed=1, shuffle=False, verbose=True)