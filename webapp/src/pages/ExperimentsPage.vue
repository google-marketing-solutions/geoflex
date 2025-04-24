<!-- <template>
  <q-page padding>
    <div class="q-pb-md flex justify-between items-center">
      <div>
        <h1 class="text-h4 q-my-none">Experiments</h1>
        <p class="text-body1 q-my-sm">Explore designs for your geo experiments.</p>
      </div>
    </div>
  </q-page>
</template>
<script setup lang="ts"></script> -->
<template>
  <q-page padding>
    <div class="q-pa-md">
      <div class="text-h4 q-mb-md">Experiments</div>

      <!-- Main card structure with tabs -->
      <q-card class="my-card">
        <q-tabs
          v-model="activeTab"
          class="text-primary"
          active-color="primary"
          indicator-color="primary"
          align="justify"
        >
          <q-tab name="datasource" label="Data Source" />
          <q-tab name="constraints" label="Constraints & Parameters" />
          <q-tab
            name="designs"
            label="Test Designs"
            :disable="!selectedDataSource || !isExplored"
          />
        </q-tabs>

        <q-separator />

        <q-tab-panels v-model="activeTab" animated>
          <!-- Data Source Tab -->
          <q-tab-panel name="datasource">
            <div class="text-h6 q-mb-md">Select Data Source</div>

            <div class="row q-col-gutter-md">
              <div class="col-12">
                <q-card class="q-pa-md">
                  <div class="text-subtitle1 q-mb-sm">Available Data Sources</div>

                  <q-select
                    v-model="selectedDataSource"
                    :options="dataSourceOptions"
                    option-label="name"
                    option-value="id"
                    label="Choose a data source"
                    outlined
                    :loading="dataSourcesStore.loading"
                    @popup-show="loadDataSourcesOnOpen"
                    @update:model-value="handleDataSourceChange"
                  >
                    <template v-slot:option="scope">
                      <q-item v-bind="scope.itemProps">
                        <q-item-section>
                          <q-item-label>{{ scope.opt.name }}</q-item-label>
                          <q-item-label caption>{{
                            scope.opt.description || 'No description provided'
                          }}</q-item-label>
                        </q-item-section>
                        <q-item-section side>
                          <q-badge :color="scope.opt.data ? 'data' : 'yellow'"> </q-badge>
                        </q-item-section>
                      </q-item>
                    </template>
                    <template v-slot:no-option>
                      <q-item>
                        <q-item-section class="text-grey">
                          {{
                            dataSourcesStore.loading
                              ? 'Loading data sources...'
                              : 'No data sources found'
                          }}
                        </q-item-section>
                      </q-item>
                    </template>
                  </q-select>

                  <div class="text-caption q-mt-sm" v-if="!selectedDataSource">
                    Select a data source with geo units, date, and conversion metrics to continue
                  </div>
                </q-card>
              </div>
            </div>

            <q-card v-if="selectedDataSource && dataSourceLoaded" class="q-mt-lg q-pa-md">
              <div class="text-subtitle1 q-mb-sm">Data Source Preview</div>
              <div class="column q-gutter-y-md">
                <div>
                  <div class="text-weight-medium">Data Columns Configuration</div>
                  <div class="row q-col-gutter-md q-mt-sm">
                    <div class="col-12 col-md-3">
                      <q-input
                        v-model="selectedDataSource.columns.geoColumn"
                        label="Geo Unit Column"
                        outlined
                        dense
                        readonly
                      />
                    </div>
                    <div class="col-12 col-md-3">
                      <q-input
                        v-model="selectedDataSource.columns.dateColumn"
                        label="Date Column"
                        outlined
                        dense
                        readonly
                      />
                    </div>
                    <div class="col-12 col-md-3">
                      <q-select
                        v-model="selectedMetric"
                        :options="selectedDataSource.columns.metricColumns"
                        label="Conversion Metric"
                        outlined
                        dense
                        hint="Select metric to use for balancing"
                      />
                    </div>
                    <div class="col-12 col-md-3">
                      <q-input
                        v-model="selectedDataSource.columns.costColumn"
                        label="Cost Column"
                        outlined
                        dense
                        readonly
                        :disable="!selectedDataSource.columns.costColumn"
                      />
                    </div>
                  </div>
                </div>

                <q-separator />

                <div>
                  <div class="text-weight-medium">Data Sample</div>
                  <q-table
                    :rows="dataSample.rows"
                    :columns="dataSample.columns"
                    row-key="id"
                    dense
                    :pagination="{ rowsPerPage: 5 }"
                  />
                </div>

                <div v-if="selectedDataSource.data?.geoUnits?.length">
                  <div class="text-weight-medium">Data Overview</div>
                  <div class="row q-col-gutter-md">
                    <div class="col-12 col-md-4">
                      <q-card class="q-pa-sm">
                        <div class="text-subtitle2">Geo Units</div>
                        <div class="text-h5">{{ selectedDataSource.data.geoUnits.length }}</div>
                      </q-card>
                    </div>
                    <div class="col-12 col-md-4">
                      <q-card class="q-pa-sm">
                        <div class="text-subtitle2">Date Range</div>
                        <div class="text-h5">{{ datePeriod }}</div>
                      </q-card>
                    </div>
                    <div class="col-12 col-md-4">
                      <q-card class="q-pa-sm">
                        <div class="text-subtitle2">{{ selectedMetric }}</div>
                        <div class="text-h5">{{ metricRange }}</div>
                      </q-card>
                    </div>
                  </div>
                </div>

                <div v-if="selectedDataSource.data?.geoUnits?.length" class="q-mt-md">
                  <q-separator class="q-my-md" />

                  <div class="text-weight-medium">Geo Units Assignment</div>
                  <div class="row items-center q-mb-md">
                    <div class="col">
                      <div class="text-body2">
                        Assign geo units to specific groups or exclude them from the test.
                        <span class="text-caption"
                          >Creating multiple test groups will automatically enable multi-cell
                          testing.</span
                        >
                      </div>
                    </div>
                    <div class="col-auto">
                      <q-input
                        v-model="geoSearch"
                        dense
                        outlined
                        placeholder="Search geo units"
                        class="q-mr-sm"
                        style="width: 200px"
                      >
                        <template v-slot:append>
                          <q-icon name="search" />
                        </template>
                      </q-input>
                    </div>
                  </div>

                  <q-table
                    :rows="filteredGeoUnits"
                    :columns="geoUnitsColumns"
                    row-key="geo"
                    dense
                    :pagination="{ rowsPerPage: 10 }"
                  >
                    <template v-slot:body-cell-assignment="props">
                      <q-td :props="props">
                        <q-select
                          v-model="props.row.assignment"
                          :options="computedAssignmentOptions"
                          dense
                          outlined
                          options-dense
                          style="min-width: 150px"
                          emit-value
                          map-options
                        >
                        </q-select>
                      </q-td>
                    </template>

                    <template v-slot:bottom="">
                      <div class="row full-width q-pa-sm">
                        <div class="col-auto q-mr-xl">
                          <q-btn
                            outline
                            color="primary"
                            icon="add_circle"
                            label="Add Test Group"
                            :disable="parameters.testGroups >= 10"
                            @click="addTestGroup"
                          />
                        </div>
                        <div
                          class="col-auto q-mr-md"
                          v-for="(count, group) in geoAssignmentCounts"
                          :key="group"
                        >
                          <q-chip
                            :color="
                              group === 'auto'
                                ? 'positive'
                                : group === 'control'
                                  ? 'blue'
                                  : group === 'exclude'
                                    ? 'negative'
                                    : 'orange'
                            "
                            text-color="white"
                            :icon="
                              group === 'control'
                                ? 'brightness_low'
                                : group === 'exclude'
                                  ? 'block'
                                  : 'science'
                            "
                          >
                            {{ formatGroupLabel(group) }}: {{ count }}
                          </q-chip>
                        </div>
                      </div>
                    </template>
                  </q-table>
                </div>
              </div>
            </q-card>
          </q-tab-panel>

          <!-- Constraints & Parameters Tab -->
          <q-tab-panel name="constraints">
            <div class="text-h6 q-mb-md">Constraints & Parameters</div>

            <q-card class="q-pa-md">
              <div class="text-subtitle1">Test Parameters and Exploration Axes</div>
              <q-separator class="q-my-md" />

              <div class="row q-mb-md">
                <div class="col">
                  <q-banner rounded class="q-mb-md">
                    <template v-slot:avatar>
                      <q-icon name="info" color="primary" />
                    </template>
                    <div class="text-body2">
                      <strong>Single value parameters</strong> act as constraints.
                      <strong>Multiple value parameters</strong> become exploration axes, generating
                      designs for all combinations. The tool will find the most powerful
                      experimental design based on your data and parameters.
                    </div>
                  </q-banner>
                </div>
              </div>

              <div class="row q-col-gutter-md">
                <!-- Left Column -->
                <div class="col-12 col-md-6">
                  <div class="text-subtitle2 q-mb-sm">Core Parameters</div>

                  <div class="q-gutter-y-md">
                    <!-- Methodology -->
                    <div>
                      <div class="row items-center">
                        <div class="col">
                          <div class="text-body2 q-mb-xs">Test Methodology</div>
                        </div>
                        <div class="col-auto">
                          <q-badge
                            v-if="
                              parameters.methodology.length > 1 ||
                              parameters.methodology.length === 0
                            "
                            color="purple"
                            label="Exploration Axis"
                          />
                          <q-badge v-else color="blue" label="Constraint" />
                        </div>
                      </div>
                      <q-option-group
                        v-model="parameters.methodology"
                        :options="methodologyOptionsDetailed"
                        type="checkbox"
                        dense
                      />
                      <div class="text-caption q-pl-md q-pt-xs">
                        <q-icon name="info" size="xs" color="grey" /> Select multiple to explore all
                        combinations
                      </div>
                    </div>

                    <!-- Structure -->
                    <div>
                      <div class="row items-center">
                        <div class="col">
                          <div class="text-body2 q-mb-xs">Test Structure</div>
                        </div>
                        <div class="col-auto">
                          <q-badge color="blue" label="Constraint" />
                        </div>
                      </div>
                      <q-option-group
                        v-model="parameters.structure"
                        :options="structureOptionsDetailed"
                        type="radio"
                        dense
                      />
                      <div v-if="parameters.structure === 'multi-cell'" class="q-mt-sm">
                        <q-input
                          v-model.number="parameters.testGroups"
                          type="number"
                          label="Number of test groups"
                          outlined
                          dense
                          min="2"
                          max="5"
                          hint="How many test variations to compare (excluding control)"
                        />
                      </div>
                    </div>

                    <!-- Optimization Target -->
                    <div>
                      <div class="row items-center">
                        <div class="col">
                          <div class="text-body2 q-mb-xs">Optimization Target</div>
                        </div>
                        <div class="col-auto">
                          <q-badge color="blue" label="Constraint" />
                        </div>
                      </div>
                      <q-option-group
                        v-model="parameters.optimizationTarget"
                        :options="[
                          { label: 'Power (find lowest MDE at specified power)', value: 'power' },
                          { label: 'MDE (find highest power at specified MDE)', value: 'mde' },
                        ]"
                        type="radio"
                        dense
                      />

                      <!-- Show power input when optimizing for power -->
                      <div v-if="parameters.optimizationTarget === 'power'" class="q-mt-sm">
                        <q-input
                          v-model.number="parameters.power"
                          type="number"
                          outlined
                          dense
                          min="50"
                          max="99"
                          step="1"
                          label="Target Power (%)"
                        >
                          <template v-slot:append>
                            <q-icon name="percent" />
                          </template>
                          <template v-slot:hint>
                            <span
                              >Probability of detecting the specified effect (typically
                              80-90%)</span
                            >
                          </template>
                        </q-input>
                      </div>

                      <!-- Show MDE input when optimizing for MDE -->
                      <div v-if="parameters.optimizationTarget === 'mde'" class="q-mt-sm">
                        <q-input
                          v-model.number="parameters.mde"
                          type="number"
                          outlined
                          dense
                          min="0.1"
                          step="0.1"
                          label="Target MDE (%)"
                        >
                          <template v-slot:append>
                            <q-icon name="percent" />
                          </template>
                          <template v-slot:hint>
                            <span>Minimum relative change to detect with specified power</span>
                          </template>
                        </q-input>
                      </div>
                    </div>

                    <!-- Significance Level -->
                    <div>
                      <div class="row items-center">
                        <div class="col">
                          <div class="text-body2 q-mb-xs">Significance Level (α)</div>
                        </div>
                        <div class="col-auto">
                          <q-badge color="blue" label="Constraint" />
                        </div>
                      </div>
                      <q-input
                        v-model.number="parameters.alpha"
                        type="number"
                        outlined
                        dense
                        min="0.01"
                        max="0.2"
                        step="0.01"
                      >
                        <template v-slot:hint>
                          <span
                            >Threshold for statistical significance (typically 0.05 or 0.1)</span
                          >
                        </template>
                      </q-input>
                    </div>
                  </div>
                </div>

                <!-- Right Column -->
                <div class="col-12 col-md-6">
                  <div class="text-subtitle2 q-mb-sm">Advanced Parameters</div>

                  <div class="q-gutter-y-md">
                    <!-- Hypothesis Type -->
                    <div>
                      <div class="row items-center">
                        <div class="col">
                          <div class="text-body2 q-mb-xs">Hypothesis Type</div>
                        </div>
                        <div class="col-auto">
                          <q-badge color="blue" label="Constraint" />
                        </div>
                      </div>
                      <q-option-group
                        v-model="parameters.hypothesisType"
                        :options="hypothesisOptionsDetailed"
                        type="radio"
                        dense
                      />
                    </div>

                    <!-- Treatment Configuration -->
                    <div>
                      <div class="row items-center">
                        <div class="col">
                          <div class="text-body2 q-mb-xs">
                            Experiment Type (a.k.a Test Treatment Configuration)
                          </div>
                        </div>
                        <div class="col-auto">
                          <q-badge color="blue" label="Constraint" />
                        </div>
                      </div>
                      <div>
                        <q-select
                          v-model="parameters.experimentType"
                          :options="treatmentOptionsDetailed"
                          outlined
                          dense
                          option-label="label"
                          option-value="value"
                          emit-value
                          map-options
                          hint="Intervention approach applied to test groups"
                        />
                      </div>
                    </div>

                    <!-- Test Duration Range -->
                    <div>
                      <div class="row items-center">
                        <div class="col">
                          <div class="text-body2 q-mb-xs">Test Duration Range (weeks)</div>
                        </div>
                        <div class="col-auto">
                          <q-badge color="purple" label="Exploration Axis" />
                        </div>
                      </div>
                      <div class="row q-col-gutter-sm">
                        <div class="col-6">
                          <q-input
                            v-model.number="parameters.durationMin"
                            type="number"
                            outlined
                            dense
                            min="1"
                            step="1"
                            label="Minimum"
                          >
                            <template v-slot:append>
                              <span class="text-caption">wks</span>
                            </template>
                          </q-input>
                        </div>
                        <div class="col-6">
                          <q-input
                            v-model.number="parameters.durationMax"
                            type="number"
                            outlined
                            dense
                            min="1"
                            step="1"
                            label="Maximum"
                          >
                            <template v-slot:append>
                              <span class="text-caption">wks</span>
                            </template>
                          </q-input>
                        </div>
                      </div>
                      <div class="text-caption q-pl-md q-pt-xs">
                        <q-icon name="info" size="xs" color="grey" /> The tool will explore test
                        durations between the minimum and maximum values
                      </div>
                    </div>

                    <!-- Pre-test Period -->
                    <div>
                      <div class="row items-center">
                        <div class="col">
                          <div class="text-body2 q-mb-xs">Pre-test Period (weeks)</div>
                        </div>
                        <div class="col-auto">
                          <q-badge color="blue" label="Constraint" />
                        </div>
                      </div>
                      <q-input
                        v-model.number="parameters.pretestPeriod"
                        type="number"
                        outlined
                        dense
                        min="0"
                        step="1"
                        hint="Historical window used for baseline metrics and matching"
                      />
                    </div>

                    <!-- Budget Allocation
                    <div>
                      <div class="row items-center">
                        <div class="col">
                          <div class="text-body2 q-mb-xs">Budget Allocation Method</div>
                        </div>
                        <div class="col-auto">
                          <q-badge color="blue" label="Constraint" />
                        </div>
                      </div>
                      <q-select
                        v-model="parameters.budgetAllocation"
                        :options="budgetOptionsDetailed"
                        outlined
                        dense
                        option-label="label"
                        option-value="value"
                        emit-value
                        map-options
                        hint="How budget is distributed across test groups"
                      />
                    </div>
                  --></div>
                </div>
              </div>

              <q-separator class="q-my-md" />

              <div class="row items-center">
                <!-- <div class="col">
                  <div class="text-body2">
                    <q-icon name="analytics" color="primary" /> Exploration will generate
                    <strong class="text-primary">{{ estimatedDesignCount }}</strong>
                    test designs based on parameter combinations.
                  </div>
                </div> -->
                <div class="col-auto">
                  <q-btn label="Reset" color="grey" flat class="q-mr-sm" @click="resetParameters" />
                  <q-btn
                    label="Run Exploration"
                    color="primary"
                    :disable="!selectedDataSource || !dataSourceLoaded"
                    @click="runExploration"
                  />
                </div>
              </div>
            </q-card>
          </q-tab-panel>

          <!-- Test Designs Tab -->
          <q-tab-panel name="designs">
            <div class="text-h6 q-mb-md">Test Designs</div>

            <q-card class="q-pa-md">
              <div class="row items-center q-mb-md">
                <div class="col">
                  <div class="text-subtitle1">Found {{ testDesigns.length }} designs</div>
                </div>
                <div class="col-auto">
                  <q-select
                    v-model="sortBy"
                    :options="sortOptions"
                    label="Sort by"
                    outlined
                    dense
                    style="min-width: 200px"
                  />
                </div>
              </div>

              <div v-for="(design, index) in sortedDesigns" :key="index" class="q-mb-lg">
                <q-card bordered>
                  <q-card-section>
                    <div class="row items-center">
                      <div class="col">
                        <div class="text-h6">Design #{{ index + 1 }}</div>
                        <div class="text-caption">
                          Power: {{ design.power }}% | MDE: {{ design.mde }}% | Duration:
                          {{ design.duration }} weeks
                        </div>
                      </div>
                      <div class="col-auto">
                        <q-btn-group flat>
                          <q-btn color="primary" icon="visibility" @click="viewDesign(design)" />
                          <q-btn color="positive" icon="download" @click="downloadDesign(design)" />
                          <!-- <q-btn color="secondary" icon="send" @click="exportDesign(design)" /> -->
                        </q-btn-group>
                      </div>
                    </div>
                  </q-card-section>

                  <q-separator />

                  <q-card-section>
                    <div class="row q-col-gutter-md">
                      <!-- Parameters Summary -->
                      <div class="col-12 col-md-6">
                        <div class="text-subtitle2">Parameters</div>
                        <q-list dense>
                          <q-item v-for="(value, key) in design.parameters" :key="key">
                            <q-item-section>
                              <q-item-label caption>{{ formatKey(key) }}</q-item-label>
                              <q-item-label>{{ value }}</q-item-label>
                            </q-item-section>
                          </q-item>
                        </q-list>
                      </div>

                      <!-- Statistical Properties -->
                      <div class="col-12 col-md-6">
                        <div class="text-subtitle2">Statistical Properties</div>
                        <div class="row q-col-gutter-md">
                          <div class="col-6">
                            <q-item dense>
                              <q-item-section>
                                <q-item-label caption>Power</q-item-label>
                                <q-item-label class="text-primary text-weight-bold"
                                  >{{ design.power.toFixed(1) }}%</q-item-label
                                >
                              </q-item-section>
                            </q-item>
                          </div>
                          <div class="col-6">
                            <q-item dense>
                              <q-item-section>
                                <q-item-label caption
                                  >MDE ({{ design.parameters.primary_metric }})</q-item-label
                                >
                                <q-item-label class="text-primary text-weight-bold"
                                  >{{ (design.mde * 100).toFixed(1) }}%</q-item-label
                                >
                              </q-item-section>
                            </q-item>
                          </div>
                        </div>
                      </div>
                    </div>

                    <!-- Groups Summary -->
                    <div class="q-mt-md">
                      <div class="text-subtitle2">Groups</div>
                      <div class="row q-col-gutter-md">
                        <div
                          v-for="(geos, groupName) in design.groups"
                          :key="groupName as string"
                          class="col-12 col-md-6"
                        >
                          <q-card flat bordered>
                            <q-card-section class="q-py-sm bg-primary text-white">
                              <div class="text-subtitle2">
                                {{ groupName }} ({{ geos.length }} geos)
                              </div>
                            </q-card-section>
                            <q-card-section class="q-pa-sm">
                              <div
                                class="geo-chips-container"
                                style="max-height: 200px; overflow-y: auto"
                              >
                                <q-chip
                                  v-for="geo in geos"
                                  :key="geo"
                                  :color="
                                    isFixedGeo(geo, groupName as string) ? 'orange' : 'primary'
                                  "
                                  text-color="white"
                                  size="sm"
                                  class="q-ma-xs"
                                >
                                  {{ geo }}
                                </q-chip>
                              </div>
                            </q-card-section>
                          </q-card>
                        </div>
                      </div>
                    </div>
                  </q-card-section>

                  <!-- Expandable section with charts -->
                  <!--
                  <q-expansion-item
                    icon="analytics"
                    label="View metrics"
                    caption="Pre-test metrics comparison"
                  >
                    <q-card-section>
                      <div class="row q-col-gutter-md">
                        <div class="col-12 col-md-6">
                          !-- Placeholder for chart component --
                          <div class="bg-grey-3 flex flex-center" style="height: 200px">
                            Group Balance Chart
                          </div>
                        </div>
                        <div class="col-12 col-md-6">
                          !-- Placeholder for chart component --
                          <div class="bg-grey-3 flex flex-center" style="height: 200px">
                            Time Series Comparison
                          </div>
                        </div>
                      </div>
                    </q-card-section>
                  </q-expansion-item>
                  -->
                </q-card>
              </div>
            </q-card>
          </q-tab-panel>
        </q-tab-panels>
      </q-card>
    </div>

    <!-- Design Detail Dialog -->
    <q-dialog v-model="designDetailDialog" maximized persistent>
      <q-card>
        <q-card-section class="row items-center">
          <div class="text-h6">Design Details</div>
          <q-space />
          <q-btn icon="close" flat round dense v-close-popup />
        </q-card-section>

        <q-separator />

        <q-card-section v-if="selectedDesign" class="q-pa-md">
          <div class="row q-col-gutter-md">
            <!-- Design metadata -->
            <div class="col-12 col-md-4">
              <q-card class="q-pa-md">
                <div class="text-subtitle1">Design Summary</div>
                <q-list dense>
                  <q-item v-for="(value, key) in selectedDesign.parameters" :key="key">
                    <q-item-section>
                      <q-item-label caption>{{ formatKey(key) }}</q-item-label>
                      <q-item-label>{{ value }}</q-item-label>
                    </q-item-section>
                  </q-item>
                </q-list>
              </q-card>
            </div>

            <!-- Statistics -->
            <div class="col-12 col-md-8">
              <q-card class="q-pa-md">
                <div class="text-subtitle1">Statistical Properties</div>
                <div class="row q-col-gutter-md">
                  <div class="col-6 col-md-3">
                    <div class="text-caption">Power</div>
                    <div class="text-h5">{{ selectedDesign.power }}%</div>
                  </div>
                  <div class="col-6 col-md-3">
                    <div class="text-caption">MDE</div>
                    <div class="text-h5">{{ selectedDesign.mde }}%</div>
                  </div>
                  <div class="col-6 col-md-3">
                    <div class="text-caption">Duration</div>
                    <div class="text-h5">{{ selectedDesign.duration }} wks</div>
                  </div>
                  <!-- <div class="col-6 col-md-3">
                    <div class="text-caption">Pre-test Period</div>
                    <div class="text-h5">{{ selectedDesign.parameters.pretestPeriod }} wks</div>
                  </div> -->
                </div>
              </q-card>

              <!-- Power charts -->
              <q-card class="q-pa-md q-mt-md">
                <div class="text-subtitle1">Power Analysis</div>
                <div class="bg-grey-3 flex flex-center q-mt-sm" style="height: 200px">
                  Power Analysis Chart
                </div>
              </q-card>
            </div>

            <!-- Group details -->
            <div class="col-12">
              <q-tabs
                v-model="groupTab"
                class="text-primary"
                active-color="primary"
                indicator-color="primary"
                align="justify"
              >
                <q-tab
                  v-for="(group, groupName) in selectedDesign.groups"
                  :key="groupName"
                  :name="groupName"
                  :label="groupName"
                />
              </q-tabs>

              <q-separator />

              <q-tab-panels v-model="groupTab" animated>
                <q-tab-panel
                  v-for="(group, groupName) in selectedDesign.groups"
                  :key="groupName"
                  :name="groupName"
                >
                  <div class="row q-col-gutter-md">
                    <div class="col-12 col-md-4">
                      <q-card class="q-pa-md">
                        <div class="text-subtitle1">{{ groupName }} Summary</div>
                        <q-list dense>
                          <q-item>
                            <q-item-section>
                              <q-item-label caption>Geo Count</q-item-label>
                              <q-item-label>{{ group.length }}</q-item-label>
                            </q-item-section>
                          </q-item>
                          <!-- <q-item>
                            <q-item-section>
                              <q-item-label caption>Total Conversions</q-item-label>
                              <q-item-label>{{ formatNumber(group.conversionTotal) }}</q-item-label>
                            </q-item-section>
                          </q-item>
                          <q-item>
                            <q-item-section>
                              <q-item-label caption>Average Per Geo</q-item-label>
                              <q-item-label>{{ formatNumber(group.conversionAvg) }}</q-item-label>
                            </q-item-section>
                          </q-item> -->
                        </q-list>
                      </q-card>
                    </div>

                    <div class="col-12 col-md-8">
                      <q-card class="q-pa-md">
                        <div class="text-subtitle1">Time Series</div>
                        <div class="bg-grey-3 flex flex-center q-mt-sm" style="height: 200px">
                          Time Series Chart
                        </div>
                      </q-card>
                    </div>
                  </div>

                  <q-card class="q-pa-md q-mt-md">
                    <div class="text-subtitle1">Geo Units</div>
                    <q-table
                      :rows="group"
                      :columns="geoDetailColumns"
                      row-key="geo"
                      dense
                      :pagination="{ rowsPerPage: 10 }"
                    />
                  </q-card>
                </q-tab-panel>
              </q-tab-panels>
            </div>
          </div>
        </q-card-section>

        <q-card-actions align="right">
          <q-btn color="primary" label="Download" @click="downloadDesign(selectedDesign)" />
          <!-- <q-btn
            color="secondary"
            label="Export to Google Ads"
            @click="exportDesign(selectedDesign)"
          /> -->
          <q-btn color="grey" label="Close" v-close-popup />
        </q-card-actions>
      </q-card>
    </q-dialog>
  </q-page>
</template>

<script setup lang="ts">
import { ref, computed, reactive, watch } from 'vue';
import { useDataSourcesStore, type DataSource } from 'src/stores/datasources';
import type { QTableColumn } from 'quasar';
import { postApiUi } from 'src/boot/axios';
import type { ExperimentDesign, ExperimentExploreResponse } from 'src/components/models';

const dataSourcesStore = useDataSourcesStore();

// Tab state
const activeTab = ref('datasource');

// Data source state
const selectedDataSource = ref<DataSource | null>(null);
const dataSourceLoaded = ref(false);
const selectedMetric = ref('');

watch(selectedMetric, (newValue) => {
  if (geoUnitsColumns.length > 1) {
    geoUnitsColumns[1].label = `Avg. ${newValue || 'Conversions'}`;
  }
});

// Derived data source properties
const dataSourceOptions = computed(() => dataSourcesStore.datasources);

// Function to load data sources when dropdown is opened
async function loadDataSourcesOnOpen() {
  if (!dataSourcesStore.isLoaded && !dataSourcesStore.loading) {
    await dataSourcesStore.loadDataSources();
  }
}

// Data source handling
const handleDataSourceChange = async (dataSource) => {
  if (!dataSource) {
    dataSourceLoaded.value = false;
    return;
  }

  try {
    if (!dataSource.data) {
      await dataSourcesStore.loadDataSourceData(dataSource);
    }

    dataSourceLoaded.value = true;

    // Set default metric
    if (dataSource.columns.metricColumns.length > 0) {
      selectedMetric.value = dataSource.columns.metricColumns[0];
    }

    // Populate geo units
    if (dataSource.data && dataSource.data.geoUnits) {
      populateGeoUnits(dataSource);
    }
  } catch (error) {
    console.error('Error loading data source:', error);
  }
};

function populateGeoUnits(dataSource) {
  const { geoColumn } = dataSource.columns;
  const geoGroups = {};
  dataSource.data.rawRows.forEach((row) => {
    const geo = row[geoColumn];
    geoGroups[geo] = {};
  });
  geoUnits.value = Object.keys(geoGroups).map((geo) => {
    return {
      geo,
      assignment: 'auto',
    };
  });
}

const datePeriod = computed(() => {
  if (!selectedDataSource.value?.data?.uniqueDates?.length) return 'No dates';
  const dates = selectedDataSource.value.data.uniqueDates;
  return `${dates[0]} - ${dates[dates.length - 1]}`;
});

const metricRange = computed(() => {
  if (!selectedDataSource.value?.data?.metrics || !selectedMetric.value) return 'N/A';

  const metric = selectedDataSource.value.data.metrics.find((m) => m.name === selectedMetric.value);
  if (!metric) return 'N/A';

  return `${formatNumber(metric.min)} - ${formatNumber(metric.max)}`;
});

// Data sample for preview
const dataSample = computed(() => {
  if (!selectedDataSource.value || !selectedDataSource.value.data?.rawRows) {
    return {
      columns: [] as QTableColumn[],
      rows: [],
    };
  }

  const rawRows = selectedDataSource.value.data.rawRows;
  const sampleRows = rawRows.slice(0, 100).map((row, index) => ({ id: index, ...row }));

  // Extract column definitions from the first row
  const columns = Object.keys(rawRows[0] || {}).map((key) => ({
    name: key,
    label: key,
    field: key,
    align: typeof rawRows[0][key] === 'number' ? 'right' : 'left',
  }));

  return {
    columns: columns as QTableColumn[],
    rows: sampleRows,
  };
});

const geoSearch = ref('');
const geoUnits = ref([]);

// Geo units table
const geoUnitsColumns: QTableColumn[] = [
  { name: 'geo', label: 'Geo Unit', field: 'geo', align: 'left' },
  { name: 'assignment', label: 'Assignment', field: 'assignment', align: 'center' },
];

// Computed properties for geo assignment
const computedAssignmentOptions = computed(() => {
  const options = [
    { label: 'Auto', value: 'auto' },
    { label: 'Control Group', value: 'control' },
    { label: 'Test Group', value: 'test' },
  ];

  // Add additional test groups if needed
  for (let i = 1; i < parameters.testGroups; i++) {
    options.push({ label: `Test Group ${String.fromCharCode(65 + i)}`, value: `test-${i + 1}` });
  }

  options.push({ label: 'Exclude', value: 'exclude' });

  return options;
});

const filteredGeoUnits = computed(() => {
  if (!geoSearch.value) return geoUnits.value;

  const query = geoSearch.value.toLowerCase();
  return geoUnits.value.filter((unit) => unit.geo.toLowerCase().includes(query));
});

// return counts of each group to show badges
const geoAssignmentCounts = computed(() => {
  const counts = {
    auto: 0,
    control: 0,
    test: 0,
    exclude: 0,
  };

  // Initialize additional test groups
  for (let i = 1; i < parameters.testGroups; i++) {
    counts[`test-${i + 1}`] = 0;
  }

  // Count assignments
  geoUnits.value.forEach((unit) => {
    if (counts[unit.assignment] !== undefined) {
      counts[unit.assignment]++;
    } else {
      counts.auto++;
    }
  });

  return counts;
});

function addTestGroup() {
  parameters.structure = 'multi-cell';
  parameters.testGroups++;
}

function formatGroupLabel(group) {
  if (group === 'auto') return 'Auto';
  if (group === 'control') return 'Control';
  if (group === 'test') return 'Test';
  if (group === 'exclude') return 'Excluded';
  if (group.startsWith('test-')) {
    const groupNum = parseInt(group.split('-')[1]);
    return `Test ${String.fromCharCode(64 + groupNum)}`;
  }
  return group;
}

const methodologyOptionsDetailed = [
  {
    label: 'Time Based Regression with Matched Markets (TBR-MM)',
    value: 'TBR-MM',
    desc: 'Regression-based methodology that uses time series data with matched markets',
  },
  {
    label: 'Time Based Regression with Random Assignment (TBR)',
    value: 'TBR',
    desc: 'Regression-based methodology that uses time series data with random geo assignment',
  },
  {
    label: 'Trimmed Match (TM)',
    value: 'TM',
    desc: 'Matches geo units based on similarity and trims outliers',
  },
  {
    label: 'Geo Based Regression (GBR)',
    value: 'GBR',
    desc: 'The original and simplest geo experiment methodology',
  },
];

const structureOptionsDetailed = [
  {
    label: 'Single-cell (A/B)',
    value: 'single-cell',
    desc: 'Standard A/B test with one control and one test group',
  },
  {
    label: 'Multi-cell (A/B/C...)',
    value: 'multi-cell',
    desc: 'Multiple test groups each receiving different variations',
  },
];

const hypothesisOptionsDetailed = [
  {
    label: 'One-sided (recommended)',
    value: 'one-sided',
    desc: 'Examines effect in one direction only (e.g., whether treatment performs better than control)',
  },
  {
    label: 'Two-sided',
    value: 'two-sided',
    desc: 'Examines effect in both directions (better or worse)',
  },
];

const treatmentOptionsDetailed = [
  {
    label: 'Holdback (new campaign)',
    value: 'hold_back',
    desc: 'Withhold advertising from control to measure true incremental impact',
  },
  {
    label: 'Go Dark (existing campaign)',
    value: 'go_dark',
    desc: 'Stop or reduce advertising in selected areas to determine baseline performance',
  },
  {
    label: 'Heavy Up (existing campaign)',
    value: 'heavy_up',
    desc: 'Increase advertising in test units to measure response to intensified media pressure',
  },
];

/*
const budgetOptionsDetailed = [
  {
    label: 'Equal across regions',
    value: 'equal',
    desc: 'Same budget allocated to each geo unit regardless of size',
  },
  {
    label: 'Proportional to market size',
    value: 'proportional',
    desc: 'Budget allocated in proportion to the size or importance of each geo',
  },
  {
    label: 'Optimized based on historical data',
    value: 'optimized',
    desc: 'Budget allocated based on historical performance data',
  },
];
*/

const DEFAULT_PARAMETERS = {
  // Core parameters
  methodology: [] as string[], // Empty array means "explore all methodologies"
  structure: 'single-cell',
  testGroups: 1, // Only used when structure is 'multi-cell'

  // Optimization target
  optimizationTarget: 'power', // 'power' or 'mde'

  // Power settings
  power: 80, // Target power when optimizationTarget is 'power'
  // MDE settings
  mde: 5, // Target MDE when optimizationTarget is 'mde'

  // Duration settings
  durationMin: 4,
  durationMax: 8,

  experimentType: 'hold_back',
  alpha: 0.05,
  hypothesisType: 'one-sided',
  pretestPeriod: 4,
  //budgetAllocation: 'equal',
};

// Initialize parameters with default values
const parameters = reactive({ ...DEFAULT_PARAMETERS });

// Watch for changes in testGroups count
watch(
  () => parameters.testGroups,
  (newCount, oldCount) => {
    // If we reduced the number of test groups
    if (newCount < oldCount) {
      // Find geo units assigned to test groups that no longer exist
      geoUnits.value.forEach((unit) => {
        // Check if this is a test group assignment (test-2, test-3, etc)
        if (unit.assignment.startsWith('test-')) {
          // Extract the group number (test-2 -> 2)
          const groupNum = parseInt(unit.assignment.split('-')[1]);

          // If this group number is higher than our current count, reset to auto
          // +1 because test groups are 1-indexed
          if (groupNum > newCount) {
            unit.assignment = 'auto';
            console.log(`Reset ${unit.geo} from ${unit.assignment} to auto`);
          }
        }
      });
    }
  },
);

// Watch for changes in structure type
watch(
  () => parameters.structure,
  (newStructure) => {
    if (newStructure === 'single-cell') {
      // When switching to single-cell, set testGroups to 1
      parameters.testGroups = 1;

      // Reset any geo units assigned to test-2, test-3, etc. to 'test'
      geoUnits.value.forEach((unit) => {
        if (unit.assignment.startsWith('test-') && unit.assignment !== 'test') {
          // Convert any test-2, test-3, etc. to just 'test'
          unit.assignment = 'test';
        }
      });
    } else if (newStructure === 'multi-cell' && parameters.testGroups < 2) {
      // When switching to multi-cell, ensure we have at least 2 test groups
      parameters.testGroups = 2;
    }
  },
);

const resetParameters = () => {
  // Reset parameters to default values
  Object.assign(parameters, DEFAULT_PARAMETERS);
};

const sortOptions = [
  { label: 'Highest Power', value: 'power' },
  { label: 'Lowest MDE', value: 'mde' },
  { label: 'Shortest Duration', value: 'duration' },
];

// Test designs
const isExplored = ref(false);
const testDesigns = ref([] as ExperimentDesign[]);
const sortBy = ref(sortOptions[0]);
const sortedDesigns = computed<ExperimentDesign[]>(() => {
  const designs = [...testDesigns.value];
  const field = sortBy.value.value;

  if (field === 'power') {
    return designs.sort((a, b) => b.power - a.power);
  } else if (field === 'duration') {
    return designs.sort((a, b) => a.duration - b.duration);
  }

  return designs;
});

// Design detail dialog
const designDetailDialog = ref(false);
const selectedDesign = ref<ExperimentDesign>(null);
const groupTab = ref('Control');
const geoDetailColumns: QTableColumn[] = [
  { name: 'geo', label: 'Geo Unit', field: (row) => row, align: 'left' },
];

// Format helpers
const formatNumber = (num) => {
  return new Intl.NumberFormat().format(num);
};

const formatKey = (key) => {
  return key
    .replace(/([A-Z])/g, ' $1') // Insert space before capital letters
    .replace(/^./, (str) => str.toUpperCase()); // Capitalize first letter
};

let lastRequest;

async function runExploration() {
  // TODO:
  const request = {
    // Datasource ID from the selected datasource
    datasource_id: selectedDataSource.value?.id,

    // Core parameters
    experiment_type: parameters.experimentType,
    primary_metric: selectedMetric.value,
    //secondary_metrics: [],

    // Test parameters
    n_cells: parameters.structure === 'multi-cell' ? parameters.testGroups : 2,
    alpha: parameters.alpha,
    alternative_hypothesis: parameters.hypothesisType,

    // Duration parameters
    min_runtime_weeks: parameters.durationMin,
    max_runtime_weeks: parameters.durationMax,

    // Methodology options (empty means explore all)
    methodologies: parameters.methodology.length > 0 ? parameters.methodology : [],

    // Optimization target
    optimization_target: parameters.optimizationTarget,
    target_power: parameters.optimizationTarget === 'power' ? parameters.power : null,
    target_mde: parameters.optimizationTarget === 'mde' ? parameters.mde : null,

    // Geo assignments from user selections
    fixed_geos: getGeoAssignments(geoUnits.value),

    pretest_weeks: parameters.pretestPeriod,
    // TODO:
    //trimming_quantile_candidates: List[float] = [0.0]
  };
  lastRequest = request;

  const response = await postApiUi<ExperimentExploreResponse>(
    'experiments/explore',
    request,
    'Running exploration',
  );
  if (!response) return;
  // Process the response
  testDesigns.value = response.data.designs;
  isExplored.value = true;

  // Navigate to designs tab
  activeTab.value = 'designs';
}

// Helper function to format geo assignments from UI selections
function getGeoAssignments(geoUnits) {
  // Start with empty groups
  const assignments = {
    control: [],
    treatment: [[]], // Start with one treatment group for single-cell
    exclude: [],
  };

  // Process multi-cell test if needed
  if (parameters.structure === 'multi-cell' && parameters.testGroups > 1) {
    // Initialize the right number of treatment groups
    assignments.treatment = Array.from({ length: parameters.testGroups }, () => []);
  }

  // Process each geo unit
  geoUnits.forEach((unit) => {
    if (unit.assignment === 'control') {
      assignments.control.push(unit.geo);
    } else if (unit.assignment === 'exclude') {
      assignments.exclude.push(unit.geo);
    } else if (unit.assignment === 'test') {
      // Default test group (group 0)
      assignments.treatment[0].push(unit.geo);
    } else if (unit.assignment.startsWith('test-')) {
      // Handle test-2, test-3, etc. for multi-cell
      const groupIndex = parseInt(unit.assignment.split('-')[1]) - 1;
      if (groupIndex >= 0 && groupIndex < assignments.treatment.length) {
        assignments.treatment[groupIndex].push(unit.geo);
      }
    }
  });

  return assignments;
}

function downloadDesign(design) {
  // TODO:
  // Create a design export object
  const exportData = {
    name: `Geo Test Design ${new Date().toISOString().split('T')[0]}`,
    source: selectedDataSource.value?.name,
    metric: selectedMetric.value,
    parameters: design.parameters,
    statistics: {
      power: design.power,
      mde: design.mde,
      duration: design.duration,
    },
    groups: {},
  };

  // Add geo assignments for each group
  Object.keys(design.groups).forEach((groupName) => {
    exportData.groups[groupName] = design.groups[groupName].geos.map((geo) => ({
      geo: geo.geo,
    }));
  });
  // const filename = `geo-test-design-${Date.now()}.json`;
  // TODO: send to the server
}

function viewDesign(design) {
  selectedDesign.value = design;
  groupTab.value = 'Control'; // Reset to first tab
  designDetailDialog.value = true;
}

// Helper function to check if a geo was fixed in the assignment
function isFixedGeo(geo: string, groupName: string): boolean {
  if (!lastRequest.fixed_geos) return false;

  if (groupName === 'Control' && lastRequest.fixed_geos.control?.includes(geo)) {
    return true;
  }

  if (groupName.startsWith('Test')) {
    // For multi-cell tests, need to check the right treatment group
    const groupIndex =
      groupName === 'Test' ? 0 : Number(groupName.replace('Test ', '').charCodeAt(0) - 65); // 'Test A' -> 0, 'Test B' -> 1

    if (
      groupIndex >= 0 &&
      lastRequest.fixed_geos.treatment &&
      lastRequest.fixed_geos.treatment[groupIndex]?.includes(geo)
    ) {
      return true;
    }
  }

  return false;
}
</script>
