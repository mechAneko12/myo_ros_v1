# Ubuntu18.04LTSにros-melodicをインストールしてHRI-Hand_ROSを動かす方法

最初に`sudo su -`すれば、rootでできるのでsudoいらない

## ***install python2.7***
> ```
> $ sudo apt install python
> ```
## ***install ros-melodic***
> ```
> $ sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
> $ sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
> $ sudo apt update
> ```
> ここでつまずいたら、
> ```
> $ sudo apt-key del F42ED6FBAB17C654
> $ sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
> $ sudo apt update
> ```
> ```
> $ sudo apt install ros-melodic-desktop-full
> ```
> ROSの依存関係を解決してくれるrosdepをインストール、初期化。
> ```
> $ sudo apt install python-rosdep
> $ sudo rosdep init
> $ rosdep update
> ```
> 環境変数の設定。~/.bashrcというターミナルの設定ファイルにROS関係の設定を追記。
> ```
> $ echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
> $ source ~/.bashrc
> ```
> buildするためにあらかじめcatkinをインストール。
> ```
> $ sudo apt install python-catkin-tools
> ```
> 必要に応じてツールをインストール。
> ```
> $ sudo apt install python-rosinstall python-rosinstall-generator python-wstool build-essential
> ```
## ***catkinワークスペースの作成、初期化(`catkin_ws`は任意の名前)***
> ```
> $ mkdir -p ~/catkin_ws/src
> $ cd ~/catkin_ws/
> $ catkin build
> $ catkin_init_workspace
> ```
> `catkin build`をしないと`~/catkin_ws/log`が生成されずにあとで`catkin build`するときに怒られる。(catkin_init_workspaceはいらなさそう？)
> 
> ワークスペースの場所を設定ファイルに書いて教えてあげる。
> ```
> $ echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
> $ source ~/.bashrc
> ```
> (`~/catkin_ws`で`source devel/setup.bash`するのと同じ。何度もやらずに済む)

## ***HRI-Hand-ROSの導入***
> githubに書いてあるとおり、
> ```
> $ cd ~/catkin_ws/src && git clone https://github.com/MrLacuqer/HRI-Hand-ROS.git
> $ cd ~/catkin_ws
> $ catkin build
> ```
> ```
> $ rospack profile && rosstack profile
> ```
> ２つのターミナルを開いておいて、どちらにも`source ~/.bashrc`する。(どちらにも`source ~/catkin/devel/setup.bash`するのと同じ)
> 一方で、
> ```
> $ roslaunch hri_hand_control hri_hand_control.launch
> ```
> もう一方で、
> ```
> $ rosrun hri_hand_control hri_joint_state_pub.py
> ```

## ***起こりうるエラー***
> - ### `~/hard_ws/src/HRI-Hand-ROS/hri_hand_control/script`内のpythonファイルに対するpermission error
>     ```
>     $ sudo chmod u+x {}.py
>     ```
> - ### serialに関するエラー
>     `~/catkin_ws/src/HRI-Hand-ROS/hri_hand_control/script/hri_joint_state_pub.py`のstmserをコメントアウト
> 
> - ### pathが通っていいない
>     `RLException: [hri_hand_control.launch] is neither a launch file in package [hri_hand_control] nor is [hri_hand_control] a launch file name
The traceback for the exception was written to the log file` <br>
>     これに対しては
>     ```
>     $ source ~/catkin_ws/devel/setup.bash
>     ```
> - ### `/home/usr/.ros/log`などへのpermission error
>     `.ros`に鍵がかかっていたら、
>     ```
>     $ rosdep fix-permissions
>     ```

## ***Appendix***
> - pathの確認
>     ```
>     $ echo $ROS_PACKAGE_PATH
>     /home/youruser/catkin_ws/src:/opt/ros/kinetic/share
>     ```
> - マイコンの接続とArduino IDE
>    ```
>    $ ls -l /dev/serial/by-id/
>    合計 0
>    lrwxrwxrwx 1 root root 13 11月 21 18:01 usb-2341_0043-if00 -> ../../ttyACM0
>    ```
>    で割り振られたポートを確認。
>    ```
>    sudo chmod a+rw /dev/ttyACM0
>    ```
> - virtualboxを使うとき
>    ```
>    $ sudo chmod 666 /dev/ttyACM0 (/dev/USB0)
>    ```
>    してからシリアルポート有効化してvirtualbox起動。

## ***ROS with Myo***
>   ```
>   $ ls -l /dev/serial/by-id/
>   ```
>   アドレスがあっているか確認した後、
>   sudo chmod u+x /dev/ttyACM?
>   PyoManager.pyc <-いらない
>   launch
>   hri_joint_state_pub.py