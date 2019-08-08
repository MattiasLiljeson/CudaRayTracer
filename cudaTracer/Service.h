#pragma once

#include <string>
#include <map>
#include <vector>
#include <iostream>

class Service {
   public:
    virtual void update(float dt) = 0;
};

class ServiceRegistry {
    std::vector<Service*> list;
    std::map<std::string, Service*> map;

    ServiceRegistry() {}

   public:
    static ServiceRegistry& instance() {
        static ServiceRegistry instance;  // Guaranteed to be destroyed.
                                          // Instantiated on first use.
        return instance;
    }
    ServiceRegistry(ServiceRegistry const&) = delete;
    void operator=(ServiceRegistry const&) = delete;

    template <typename T>
    void add(T* service) {
        std::string name = typeid(T).name();
        add(name, service);
    }

    void add(std::string name, Service* service) {
        map.insert({name, service});
        list.push_back(service);
    }

    template <typename T>
    T* get() {
        std::string name = typeid(T).name();
        return static_cast<T*>(get(name));
    }

    Service* get(std::string name) {
        auto it = map.find(name);
        if (it != map.end()) {
            // element found;
            return map.at(name);
        }
        return nullptr; // TODO: replace with optional
    }

    void updateAll(float dt) {
        for (const auto& e : list) {
            e->update(dt);
        }
    }
};