<!--
 Copyright 2025 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

<template>
  <q-page class="row items-center justify-evenly">
    <div>
      <h3>Welcome to GeoFlex!</h3>
      <p>This is your one-stop solution for conducting Geo experiments with Ads.</p>
      <p>
        The first and foremost prerequisite to get started is up get your business metrics
        attributed by geo unit and date. Once you get it, create a
        <router-link to="/datasources" class="text-primary" style="text-decoration: underline"
          >data source</router-link
        >
        and start exploring possible experiment designs in
        <router-link to="/experiments" class="text-primary" style="text-decoration: underline"
          >Experiments</router-link
        >.
      </p>
      <q-btn to="/configuration" label="Go to Configuration" color="primary" />
    </div>
  </q-page>
</template>

<script setup lang="ts">
//import { ref } from 'vue';

defineOptions({
  name: 'IndexPage',
});
</script>
