#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/catkin_ws/src/uni_lace/uni_lace"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/catkin_ws/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/catkin_ws/install/lib/python3/dist-packages:/catkin_ws/build/uni_lace/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/catkin_ws/build/uni_lace" \
    "/usr/bin/python3" \
    "/catkin_ws/src/uni_lace/uni_lace/setup.py" \
    egg_info --egg-base /catkin_ws/build/uni_lace \
    build --build-base "/catkin_ws/build/uni_lace" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/catkin_ws/install" --install-scripts="/catkin_ws/install/bin"
