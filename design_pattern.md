# 设计模式
设计模式（Design pattern）代表了最佳的实践，是软件开发人员在软件开发过程中面临的一般问题的解决方案。这些解决方案是众多软件开发人员不断的试验和错误总结出来的。使用设计模式是为了重用代码、让代码更容易被他人理解、保证代码可靠性。设计模式是软件工程的基石。
四人帮首次提出设计模式，他们提出的设计模式主要是基于以下的面向对象设计原则：
+ 对接口编程而不是对实现编程
+ 优先使用对象组合而不是继承

设计模式有23种，可以分为三类：
+ 创建型模式
  - 工厂方法模式
  - 抽象工厂模式
  - 单例模式
  - 创造者模式
  - 原型模式
+ 结构型模式
  - 适配器模式
  - 装饰器模式
  - 代理模式
  - 外观模式
  - 桥接模式
  - 组合模式
  - 享元模式
+ 行为型模式
  - 策略模式
  - 模板方法模式
  - 观察者模式
  - 迭代器模式
  - 责任链模式
  - 命令模式
  - 备忘录模式
  - 状态模式
  - 访问者模式
  - 中介者模式
  - 解释器模式

## 设计模式的六大原则
1. 开闭原则：对扩展开放，对修改关闭。简言之，是为了使程序的扩展性好，易于维护和升级。
2. 里氏代换原则：任何基类可以实现的地方，子类一定可以实现
3. 依赖倒转原则：针对接口编程，依赖于抽象而不依赖与具体
4. 接口隔离原则：使用多个隔离的接口，比使用单个接口要好。也就是降低类之间的耦合度
5. 迪米特原则（最少知道原则）：一个实体应当尽量少地与其他实体之间发生相互作用，使得系统功能模块相对独立
6. 合成服用原则：尽量使用合成/聚合的方式，而不是使用继承

---

## 工厂方法模式
意图：定义一个创建对象的接口，让其子类自己决定实例化哪一个工厂类，工厂模式使其创建过程延迟到子类进行
#### code:
```
class Person:
    def __init__(self):
        self.name = None
        self.gender = None

    def get_name(self):
        return self.name

    def get_gender(self):
        return self.gender


class Male(Person):
    def __init__(self, name):
        print("hello Mr." + name)


class Female(Person):
    def __init__(self, name):
        print("hello Miss." + name)


class Factory:
    def get_person(self, name, gender):
        if gender == "M":
            return Male(name)
        if gender == "F":
            return Female(name)


if __name__ == "__main__":
    factory = Factory()
    person = factory.get_person("ZMZ", "M")
```

---

## 抽象工厂模式
提供一个创建一系列相关或相互依赖对象的接口，而无需指定它们具体的类
#### code:
```
from abc import abstractmethod, ABCMeta


# ------抽象产品------
class PhoneShell(metaclass=ABCMeta):
    @abstractmethod
    def show_shell(self):
        pass


class CPU(metaclass=ABCMeta):
    @abstractmethod
    def show_cpu(self):
        pass


class OS(metaclass=ABCMeta):
    def show_os(self):
        pass


# ------抽象工厂------
class PhoneFactory(metaclass=ABCMeta):
    @abstractmethod
    def make_shell(self):
        pass

    @abstractmethod
    def make_cpu(self):
        pass

    @abstractmethod
    def make_os(self):
        pass


# ------具体产品------
class SmallShell(PhoneShell):
    def show_shell(self):
        print("this is small shell")


class BigShell(PhoneShell):
    def show_shell(self):
        print("this is big shell")


class HuaWeiShell(PhoneShell):
    def show_shell(self):
        print("this is HuaWei shell")


class QiLinCPU(CPU):
    def show_cpu(self):
        print("this is QiLin CPU")


class SnapDragonCPU(CPU):
    def show_cpu(self):
        print("this is XiaoLong CPU")


class Android(OS):
    def show_os(self):
        print("this is Android OS")


class IOS(OS):
    def show_os(self):
        print("this is IOS OS")


# ------具体工厂------
class HuaWeiFactory(PhoneFactory):
    def make_cpu(self):
        return QiLinCPU()

    def make_os(self):
        return Android()

    def make_shell(self):
        return HuaWeiShell()


class IphoneFactory(PhoneFactory):
    def make_cpu(self):
        return SnapDragonCPU()

    def make_os(self):
        return IOS()

    def make_shell(self):
        return SmallShell()


# ------客户端------
class Phone:
    def __init__(self, cpu, os, shell):
        self.cpu = cpu
        self.os = os
        self.shell = shell

    def show_info(self):
        print("phone info:")
        self.cpu.show_cpu()
        self.os.show_os()
        self.shell.show_shell()


def make_phone(factory):
    cpu = factory.make_cpu()
    os = factory.make_os()
    shell = factory.make_shell()
    return Phone(cpu, os, shell)


if __name__ == "__main__":
    p1 = make_phone(HuaWeiFactory())
    p1.show_info()
```

---

## 单例模式
保证一个类仅有一个实例，并提供一个访问它的全局访问点
#### code:
```
class Singleton(object):
    def __init__(self):
        pass

    def __new__(cls, *args, **kwargs):
        if not hasattr(Singleton, "_instance"):
            Singleton._instance = object.__new__(cls)
        return Singleton._instance


if __name__ == "__main__":
    obj1 = Singleton()
    obj2 = Singleton()
    print(obj1)
    print(obj2)
```

---

## 创造者模式
将一个复杂的构建与其表示相分离，使得同样的构建过程可以创建不同的表示
#### code:
```
from abc import abstractmethod, ABCMeta


# ------产品------
class Player:
    def __init__(self, face=None, body=None, arm=None, leg=None):
        self.face = face
        self.body = body
        self.arm = arm
        self.leg = leg

    def __str__(self):
        return "{}, {}, {}, {}".format(self.face, self.body, self.arm, self.leg)


# ------建造者------
class PlayerBuilder(metaclass=ABCMeta):
    @abstractmethod
    def build_face(self):
        pass

    @abstractmethod
    def build_body(self):
        pass

    @abstractmethod
    def build_arm(self):
        pass

    @abstractmethod
    def build_leg(self):
        pass

    @abstractmethod
    def get_player(self):
        pass


class BeautifulWomanBuilder(PlayerBuilder):
    def __init__(self):
        self.player = Player()

    def build_face(self):
        self.player.face = "beautiful face"

    def build_body(self):
        self.player.body = "thin body"

    def build_arm(self):
        self.player.arm = "thin arm"

    def build_leg(self):
        self.player.leg = "long leg"

    def get_player(self):
        return self.player


class PlayerDirector:
    def build_player(self, builder):
        builder.build_body()
        builder.build_arm()
        builder.build_leg()
        builder.build_face()
        return builder.get_player()
        

if __name__ == "__main__":
    director = PlayerDirector()
    builder = BeautifulWomanBuilder()
    p = director.build_player(builder)
    print(p)
```

---

## 原型模式
用原型实例指定创建对象的种类，并且通过拷贝这些原型创建新的对象
#### code:
```
import copy


class Prototype:
    def __init__(self):
        self._objects = {}

    def register_object(self, name, obj):
        self._objects[name] = obj

    def unregister_object(self, name):
        del self._objects[name]

    def clone(self, name, **attr):
        obj = copy.deepcopy(self._objects.get(name))
        obj.__dict__.update(attr)
        return obj


def main():
    class A:
        def __str__(self):
            return "I am A"

    a = A()
    prototype = Prototype()
    prototype.register_object("a", a)
    b = prototype.clone("a", a=1, b=2, c=3)

    print(a)
    print(b.a, b.b, b.c)


if __name__ == "__main__":
    main()

```

---

## 适配器模式
将一个类的接口转换成客户希望的另外一个接口。适配器模式使得原本由于接口不兼容而不能一起工作的那些类可以一起工作。
#### code:
```
class Target(object):
    def request(self):
        print("normal request")


class Adaptee(object):
    def special_request(self):
        print("special request")


class Adapter(Target):
    def __init__(self):
        self.adaptee = Adaptee()

    def request(self):
        self.adaptee.special_request()


if __name__ == "__main__":
    target = Adapter()
    target.request()
```

---

## 装饰器模式
动态地给一个对象添加一些额外的职责。就增加功能来说，装饰器模式相比生成子类更为灵活
#### code:
```
import functools


def memoize(fn):
    known = dict()

    @functools.wraps(fn)
    def memoizer(*args):
        if args not in known:
            known[args] = fn(*args)
        return known[args]

    return memoizer

@memoize
def nsum(n):
    assert(n >= 0), "n must be >= 0"
    return 0 if n == 0 else n + nsum(n - 1)

@memoize
def fibonacci(n):
    assert(n >= 0), "n must be >= 0"
    return n if n in (0, 1) else fibonacci(n - 1) + fibonacci(n - 2)


if __name__ == "__main__":
    from timeit import Timer
    measure = [{"exec": "fibonacci(100)", "import": "fibonacci", "func": fibonacci},
               {"exec": "nsum(200)", "import": "nsum", "func": nsum}]

    for m in measure:
        t = Timer("{}".format(m["exec"]), "from __main__ import {}".format(m["import"]))
        print("name: {}, doc: {}, executing: {}, time: {}".format(m["func"].__name__, m["func"].__doc__,
                                                                  m["exec"], t.timeit()))
```

---

## 代理模式
为其他对象提供一种代理以控制对这个对象的访问
#### code:
```
from abc import ABCMeta, abstractmethod


class Subject(metaclass=ABCMeta):
    @abstractmethod
    def get_content(self):
        pass

    def set_content(self, content):
        pass


class RealSubject(Subject):
    def __init__(self, file_name):
        self.file_name = file_name
        print("{}'s content is".format(file_name))
        f = open(file_name)
        self.__content = f.read()
        f.close()

    def get_content(self):
        return self.__content

    def set_content(self, content):
        f = open(self.file_name, "w")
        f.write(content)
        self.__content = content
        f.close()


# ------远程代理------
class ProxyA(Subject):
    def __init__(self, file_name):
        self.subj = RealSubject(file_name)

    def get_content(self):
        return self.subj.get_content()

    def set_content(self, content):
        return self.subj.set_content(content)


# ------虚代理------
class ProxyB(Subject):
    def __init__(self, file_name):
        self.file_name = file_name
        self.subj = None

    def get_content(self):
        if not self.subj:
            self.subj = RealSubject(self.file_name)
        return self.subj.get_content()


# ------保护代理------
class ProxyC(Subject):
    def __init__(self, file_name):
        self.subj = RealSubject(file_name)

    def get_content(self):
        self.subj.get_content()

    def set_content(self, content):
        raise PermissionError


if __name__ == "__main__":
    x = ProxyB('abc.txt')
    print(x.get_content())

    file_name = "abc.txt"
    username = input("input name:")
    if username != "ZMZ":
        p = ProxyC(file_name)
    else:
        p = ProxyA(file_name)
    print(p.get_content())

```

---

## 外观模式
为子系统中的一组接口提供一个一致的界面，外观模式定义了一个高层的接口，这个接口使得这一子系统更加容易使用
#### code:
```
from enum import Enum
from abc import ABCMeta, abstractmethod


State = Enum("State", "new running sleeping restart zombie")


class User:
    pass


class Process:
    pass


class File:
    pass


class Server(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    def __str__(self):
        return self.name

    @abstractmethod
    def boot(self):
        pass

    @abstractmethod
    def kill(self, restart=True):
        pass


class FileServer(Server):
    def __init__(self):
        self.name = "FileServer"
        self.state = State.new

    def boot(self):
        print("booting the {}".format(self))
        self.state = State.running

    def kill(self, restart=True):
        print("killing {}".format(self))
        self.state = State.restart if restart else State.zombie

    def create_file(self, user, name, permissions):
        print("trying to create the file '{}' for user '{}' with permissions {}".format(name, user, permissions))


class ProcessServer(Server):
    def __init__(self):
        self.name = "ProcessServer"
        self.state = State.new

    def boot(self):
        print("booting the {}".format(self))
        self.state = State.running

    def kill(self, restart=True):
        print("killing {}".format(self))
        self.state = State.restart if restart else State.zombie

    def create_process(self, user, name):
        print("trying to create the process '{}' for user '{}'".format(name, user))


class WindowServer:
    pass


class NetworkServer:
    pass


class OperatingSystem:
    def __init__(self):
        self.fs = FileServer()
        self.ps = ProcessServer()

    def start(self):
        [i.boot() for i in (self.fs, self.ps)]

    def create_file(self, user, name, permissions):
        return self.fs.create_file(user, name, permissions)

    def create_process(self, user, name):
        return self.ps.create_process(user, name)


if __name__ == "__main__":
    os = OperatingSystem()
    os.start()
    os.create_file("foo", "hello", "-rw-r-r")
    os.create_process("bar", "ls /tmp")

```

---

## 桥接模式
将抽象部分与实现部分分离，使它们都可以独立的变化。
#### code:
```
class DrawingAPI1(object):
    def draw_cicle(self, x, y, radius):
        print('API1.circle at {}:{} radius {}'.format(x, y, radius))


class DrawingAPI2(object):
    def draw_cicle(self, x, y, radius):
        print('API2.circle at {}:{} radius {}'.format(x, y, radius))


class CirlceShape(object):
    def __init__(self, x, y, radius, drawing_api):
        self._x = x
        self._y = y
        self._radius = radius
        self._drawing_api = drawing_api

    def draw(self):
        self._drawing_api.draw_cicle(self._x, self._y, self._radius)

    def scale(self, pct):
        self._radius *= pct


if __name__ == "__main__":
    shapes = (CirlceShape(1, 2, 3, DrawingAPI1()),
              CirlceShape(5, 7, 11, DrawingAPI2()))

    for shape in shapes:
        shape.scale(2.5)
        shape.draw()
```

---

## 组合模式
将对象组合成树形结构以表示”部分-整体“的层次结构。组合模式使得用户对单个对象和组合对象的使用具有一致性
#### code:
```
from abc import ABCMeta, abstractmethod

class Graphic(metaclass=ABCMeta):
    @abstractmethod
    def draw(self):
        pass

    @abstractmethod
    def add(self, graphic):
        pass

    def get_children(self):
        pass


# ------图元------
class Point(Graphic):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self):
        print(self)

    def add(self, graphic):
        raise TypeError

    def get_children(self):
        raise TypeError

    def __str__(self):
        return "point({}, {})".format(self.x, self.y)


class Line(Graphic):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def draw(self):
        print(self)

    def add(self, graphic):
        raise TypeError

    def get_children(self):
        raise TypeError

    def __str__(self):
        return "line[{}, {}]".format(self.p1, self.p2)


class Picture(Graphic):
    def __init__(self):
        self.children = []

    def add(self, graphic):
        self.children.append(graphic)

    def get_children(self):
        return self.children

    def draw(self):
        print("------START------")
        for g in self.children:
            g.draw()
        print("------END------")


if __name__ == "__main__":
    pic1 = Picture()
    point = Point(2, 3)
    pic1.add(point)
    pic1.add(Line(Point(1, 2), Point(4, 5)))
    pic1.add(Line(Point(0, 1), Point(2, 1)))

    pic2 = Picture()
    pic2.add(Point(-2, -1))
    pic2.add(Line(Point(0, 0), Point(1, 1)))

    pic = Picture()
    pic.add(pic1)
    pic.add(pic2)

    pic.draw()
```

---

## 享元模式
运用共享技术有效地支持大量细粒度的对象
#### code:
```
import random
from enum import Enum


TreeType = Enum("TreeType", "apple_tree cherry_tree peach_tree")


class Tree:
    pool = dict()

    def __new__(cls, tree_type):
        obj = cls.pool.get(tree_type, None)
        if not obj:
            obj = object.__new__(cls)
            cls.pool[tree_type] = obj
            obj.tree_type = tree_type
        return obj

    def render(self, age, x, y):
        print('render a tree of type {} and age {} at ({}, {})'.format(self.tree_type, age, x, y))


if __name__ == "__main__":
    rnd = random.Random()
    age_min = 1
    age_max = 30
    min_point = 0
    max_point = 100
    tree_count = 0

    for _ in range(10):
        t1 = Tree(TreeType.apple_tree)
        t1.render(rnd.randint(age_min, age_max),
                  rnd.randint(min_point, max_point),
                  rnd.randint(min_point, max_point))
        tree_count += 1

    for _ in range(3):
        t2 = Tree(TreeType.cherry_tree)
        t2.render(rnd.randint(age_min, age_max),
                  rnd.randint(min_point, max_point),
                  rnd.randint(min_point, max_point))
        tree_count += 1

    for _ in range(5):
        t3 = Tree(TreeType.peach_tree)
        t3.render(rnd.randint(age_min, age_max),
                  rnd.randint(min_point, max_point),
                  rnd.randint(min_point, max_point))
        tree_count += 1

    print('trees rendered: {}'.format(tree_count))
    print('trees actually created: {}'.format(len(Tree.pool)))

    t4 = Tree(TreeType.cherry_tree)
    t5 = Tree(TreeType.cherry_tree)
    t6 = Tree(TreeType.apple_tree)
    print('{} == {}? {}'.format(id(t4), id(t5), id(t4) == id(t5)))
    print('{} == {}? {}'.format(id(t5), id(t6), id(t5) == id(t6)))
```

---

## 策略模式
定义一系列的算法，把它们一个个封装起来，并且使它们可相互替换
#### code:
```
from abc import ABCMeta, abstractmethod
import random


class Sort(metaclass=ABCMeta):
    @abstractmethod
    def sort(self):
        pass


class QuickSort(Sort):
    def quick_sort(self, data, left, right):
        if left < right:
            mid = self.partition(data, left, right)
            self.quick_sort(data, left, mid - 1)
            self.quick_sort(data, mid + 1, right)

    def partition(self, data, left, right):
        tmp = data[left]
        while left < right:
            while left < right and data[right] >= tmp:
                right -= 1
            data[left] = data[right]
            while left < right and data[left] <= tmp:
                left += 1
            data[right] = data[left]
        data[left] = tmp
        return left

    def sort(self, data):
        print("quick sort")
        return self.quick_sort(data, 0, len(data) - 1)


class MergeSort(Sort):
    def merge(self, data, low, mid, high):
        i = low
        j = mid + 1
        ltmp = []

        while i <= mid and j <= high:
            if data[i] <= data[j]:
                ltmp.append(data[i])
                i += 1
            else:
                ltmp.append(data[j])
                j += 1

        while i <= mid:
            ltmp.append(data[i])
            i += 1

        while j <= high:
            ltmp.append(data[j])
            j += 1

        data[low: high + 1] = ltmp

    def merge_sort(self, data, low, high):
        if low < high:
            mid = (low + high) // 2
            self.merge_sort(data, low, mid)
            self.merge_sort(data, mid + 1, high)
            self.merge(data, low, mid, high)

    def sort(self, data):
        print("merge sort")
        return self.merge_sort(data, 0, len(data) - 1)


class Context:
    def __init__(self, data, strategy=None):
        self.data = data
        self.strategy = strategy

    def set_strategy(self, strategy):
        self.strategy = strategy

    def do_strategy(self):
        if self.strategy:
            self.strategy.sort(self.data)
        else:
            raise TypeError


if __name__ == "__main__":
    li = list(range(100000))
    random.shuffle(li)

    context = Context(li, MergeSort())
    context.do_strategy()
    print(context.data)

    random.shuffle(context.data)

    context.set_strategy(QuickSort())
    context.do_strategy()
    print(context.data)
```

---

## 模板方法模式
定义一个操作中的算法的骨架，而将一些步骤延迟到子类中。模板方法使得子类可以不改变一个算法的结构即可重定义该算法的某些特定步骤。
#### code:
```
from abc import ABCMeta,abstractmethod


class IOHandler(metaclass=ABCMeta):
    @abstractmethod
    def open(self, name):
        pass

    @abstractmethod
    def deal(self, change):
        pass

    @abstractmethod
    def close(self):
        pass

    def process(self, name, change):
        self.open(name)
        self.deal(change)
        self.close()


class FileHandler(IOHandler):
    def open(self, name):
        self.file = open(name, "w")

    def deal(self, change):
        self.file.write(change)

    def close(self):
        self.file.close()


if __name__ == "__main__":
    f = FileHandler()
    f.process("abc.txt", "Hello World")
```

---

## 观察者模式
定义对象间的一种一对多的依赖关系，当一个对象的状态发生改变时，所有依赖于它的对象都得到通知并被自动更新
#### code:
```
from abc import ABCMeta,abstractmethod


class Observer(metaclass=ABCMeta):
    @abstractmethod
    def update(self, notice):
        pass


class Notice:
    def __init__(self):
        self.observers = []

    def attach(self, obs):
        self.observers.append(obs)

    def detach(self, obs):
        self.observers.remove(obs)

    def notify(self):
        for obj in self.observers:
            obj.update(self)


class ManagerNotice(Notice):
    def __init__(self, company_info=None):
        super().__init__()
        self.__company_info = company_info

    def detach(self, obs):
        super().detach(obs)
        obs.company_info = None

    @property
    def company_info(self):
        return self.__company_info

    @company_info.setter
    def company_info(self, info):
        self.__company_info = info
        self.notify()


class Manager(Observer):
    def __init__(self):
        self.company_info = None

    def update(self, noti):
        self.company_info = noti.company_info


if __name__ == "__main__":
    notice = ManagerNotice()

    alex = Manager()
    wusir = Manager()

    print(alex.company_info, wusir.company_info)

    notice.attach(alex)
    notice.attach(wusir)
    notice.company_info = "company runs well"
    print(alex.company_info, wusir.company_info)

    notice.company_info = "company will be listed"
    print(alex.company_info, wusir.company_info)

    notice.detach(wusir)
    print(alex.company_info, wusir.company_info)

    notice.company_info = "company is broken"
    print(alex.company_info, wusir.company_info)
```

---

## 迭代器模式
提供一种方法顺序访问一个聚合对象中的元素，而又无需暴露该对象的内部表示
#### code:
```
class LinkList:
    class Node:
        def __init__(self, item=None):
            self.item = item
            self.next = None

    class LinkListIterator:
        def __init__(self, node):
            self.node = node

        def __next__(self):
            if self.node:
                cur_node = self.node
                self.node = cur_node.next
                return cur_node.item
            else:
                raise StopIteration

        def __iter__(self):
            return self

    def __init__(self, iterable=None):
        self.head = LinkList.Node(0)
        self.tail = self.head
        self.extend(iterable)

    def append(self, obj):
        s = LinkList.Node(obj)
        self.tail.next = s
        self.tail = s
        self.head.item += 1

    def extend(self, iterable):
        for obj in iterable:
            self.append(obj)

    def __iter__(self):
        return self.LinkListIterator(self.head.next)

    def __len__(self):
        return self.head.item

    def __str__(self):
        return "<<" + ", ".join(map(str, self)) + ">>"


if __name__ == "__main__":
    li = [i for i in range(100)]
    lk = LinkList(li)
    print(lk)
```

---

## 责任链模式
避免请求发送者与接收者耦合在一起，让多个对象都有可能接收请求，将这些对象连接成一条链，并且沿着这条链传递请求，知道有对象处理它为止
#### code:
```
from abc import ABCMeta, abstractmethod


class Handler(metaclass=ABCMeta):
    @abstractmethod
    def handle_leave(self, day):
        pass


class GeneralManagerHandler(Handler):
    def handle_leave(self, day):
        if day < 10:
            print("general manager agree {} day".format(day))
            return True
        else:
            print("general manager don't agree")
            return False


class DepartmentManagerHandler(Handler):
    def __init__(self):
        self.successor = GeneralManagerHandler()

    def handle_leave(self, day):
        if day < 7:
            print("department manager agree {} day".format(day))
            return True
        else:
            print("department manager no permission")
            return self.successor.handle_leave(day)


class ProjectDirectorHandler(Handler):
    def __init__(self):
        self.successor = DepartmentManagerHandler()

    def handle_leave(self, day):
        if day < 3:
            print("project director agree {} day".format(day))
            return True
        else:
            print("project director no permission")
            return self.successor.handle_leave(day)


if __name__ == "__main__":
    day = 11
    h = ProjectDirectorHandler()
    print(h.handle_leave(day))
```

---

## 命令模式
将一个请求封装成一个对象，从而使你可以用不同的请求对客户进行参数化
#### code:
```
import os

class MoveFileCommand(object):
    def __init__(self, src, dest):
        self.src = src
        self.dest = dest

    def execute(self):
        self()

    def __call__(self):
        print('renaming {} to {}'.format(self.src, self.dest))
        os.rename(self.src, self.dest)

    def undo(self):
        print('renaming {} to {}'.format(self.dest, self.src))
        os.rename(self.dest, self.src)


if __name__ == "__main__":
    command_stack = []

    command_stack.append(MoveFileCommand("abc.txt", "bar.txt"))
    command_stack.append(MoveFileCommand("bar.txt", "baz.txt"))

    for cmd in command_stack:
        cmd.execute()

    for cmd in reversed(command_stack):
        cmd.undo()
```

---

## 备忘录模式
在不破坏封装性的前提下，捕获一个对象的内部状态，并在该对象之外保存这个状态
#### code:
```
import copy


def Memento(obj, deep=False):
    state = (copy.copy, copy.deepcopy)[bool(deep)](obj.__dict__)

    def Restore():
        obj.__dict__.clear()
        obj.__dict__.update(state)

    return Restore


class Transation:
    deep = False

    def __init__(self, *targets):
        self.targets = targets
        self.commit()

    def commit(self):
        self.states = [Memento(target, self.deep) for target in self.targets]

    def roll_back(self):
        for st in self.states:
            st()


class Transational(object):
    def __init__(self, method):
        self.method = method

    def __ge__(self, obj, T):
        def transaction(*args, **kwargs):
            state = Memento(obj)
            try:
                return self.method(obj, *args, **kwargs)
            except:
                state()
                raise

        return transaction


class NumObj(object):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.value)

    def increment(self):
        self.value += 1

    @Transational
    def do_stuff(self):
        self.value = "11111"
        self.increment()


if __name__ == "__main__":
    n = NumObj(-1)
    print(n)
    t = Transation(n)
    try:
        for i in range(3):
            n.increment()
            print(n)
        t.commit()
        print("-- commited")
        for i in range(3):
            n.increment()
            print(n)
        n.value += "x"  # will fail
        print(n)
    except:
        t.roll_back()
        print("-- roll back")
    print(n)
    print("-- now doing stuff ...")
    try:
        n.do_stuff()
    except:
        print("-> doing stuff failed")
        import traceback
        traceback.print_exc(0)
        pass
    print(n)
```

---

## 状态模式
允许对象在内部状态发生改变时改变它的行为，对象看起来好像修改了它的类
#### code:
```
class State(object):
    def scan(self):
        self.pos += 1
        if self.pos == len(self.stations):
            self.pos = 0
        print("scaning... station is ", self.stations[self.pos], self.name)


class AmState(State):
    def __init__(self, radio):
        self.radio = radio
        self.stations = ["1250", "1380", "1510"]
        self.pos = 0
        self.name = "AM"

    def toggle_amfm(self):
        print("switching to FM")
        self.radio.state = self.radio.fmstate


class FmState(State):
    def __init__(self, radio):
        self.radio = radio
        self.stations = ["81.3", "89.1", "103.9"]
        self.pos = 0
        self.name = "FM"

    def toggle_amfm(self):
        print("switching to AM")
        self.radio.state = self.radio.amstate


class Radio(object):
    def __init__(self):
        self.amstate = AmState(self)
        self.fmstate = FmState(self)
        self.state = self.amstate

    def toggle_amfm(self):
        self.state.toggle_amfm()

    def scan(self):
        self.state.scan()


if __name__ == "__main__":
    radio = Radio()
    actions = [radio.scan] * 2 + [radio.toggle_amfm] + [radio.scan] * 2
    actions = actions * 2

    for action in actions:
        action()
```

---

## 访问者模式
## 中介者模式
## 解释器模式

## 参考：
+ https://www.cnblogs.com/tangkaishou/p/9246353.html
+ https://www.cnblogs.com/taosiyu/p/11293949.html
+ https://www.cnblogs.com/Liqiongyu/p/5916710.html
+ https://www.runoob.com/design-pattern/design-pattern-tutorial.html