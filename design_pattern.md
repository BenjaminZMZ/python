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
  - 迭代子模式
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
定义一个创建对象的接口，让其子类自己决定实例化哪一个工厂类，工厂模式使其创建过程延迟到子类进行
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

## 桥接模式
## 组合模式
## 享元模式
## 策略模式
## 模板方法模式
## 观察者模式
## 迭代子模式
## 责任链模式
## 命令模式
## 备忘录模式
## 状态模式
## 访问者模式
## 中介者模式
## 解释器模式

## 参考：
+ https://www.cnblogs.com/tangkaishou/p/9246353.html
+ https://www.cnblogs.com/taosiyu/p/11293949.html
+ https://www.runoob.com/design-pattern/design-pattern-tutorial.html