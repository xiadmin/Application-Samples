import ximea.xiapi as xiapi


def main() -> int:
    cam = xiapi.Camera()
    cam.open_device()

    # TODO: implement the sample.
    print("{{binary_name}}: not yet implemented", flush=True)

    cam.close_device()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
